"""
Example of Deep Q-Network (DQN) implementation using PyTorch and Gymnasium's CartPole-v1 environment.

Implementation derived from: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

Install the Gymnasium Environment package with:

```bash
pip3 install gymnasium[classic_control]
```
"""

import gymnasium as gym
import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional


# cartpole environment from Gymnasium
env = gym.make("CartPole-v1", render_mode=None)

# pytorch with GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# manual seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# reset environment and set seeds
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = functional.relu(self.layer1(x))
        x = functional.relu(self.layer2(x))
        return self.layer3(x)


# Transition is a map representing a single transition in our environment
# It essentially maps (state, action) pairs to their (next_state, reward) result
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# ReplayMemory is a cyclic buffer of bounded size that holds the transitions observed recently. 
# It also implements a .sample() method for selecting a random batch of transitions for training.
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # random sample a batch of transitions from memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4


# Get number of actions from gym action space
n_actions = env.action_space.n
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# define number of episodes to train for, depending on whether GPU is available
if torch.cuda.is_available():
    num_episodes = 100
else:
    num_episodes = 50

# initialize policy network and target network
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

# optimizer for policy network and replay memory
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# replay memory buffer
memory = ReplayMemory(capacity=10000)

# Initialize some variables
steps_done = 0
episode_durations = []


def select_action(state):
    global steps_done
    sample = random.random()

    """
    $$
    \epsilon_t = \epsilon_{end} + (\epsilon_{start} - \epsilon_{end}) \cdot e^{-\frac{t}{\epsilon_{decay}}}
    $$
    """
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # exploit: select the action with the highest Q-value
            # pick action with the largest expected reward, from the policy network
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # explore: select a random action
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


def log_episode_stats(episode_iteration: int):
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    average_duration = ""
    # Take 100 episode averages and plot them too
    if len(episode_durations) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        means_np = means.numpy()
        average_duration = f"{means_np[episode_iteration]:.2f}"

    current_duration = f"{durations_t[-1].item()}"

    print(f"\rEpisode {episode_iteration+1}\tDuration: {current_duration}\tAverage Duration: {average_duration}", end="")


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # the result is the batch elements for non-final (valid not None) next states
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    
    # concatenate the batch elements into a single tensor
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a): the model computes Q(s_t) for all actions.
    # select the Q-value for the action that was actually taken in the replay buffer
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute $V(s_{t+1})$ for all next states.
    # For the states that are not terminal, the target_net is used to estimate the "best possible" future reward
    # For the terminal states, the next state value is $0$
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        """
        DQN: Compute the maximum Q value for the next state
        $$
        V(s_{t+1}) = \max_{a} Q(s_{t+1}, a)
        $$
        """
        
        # DQN: use the target network to compute the maximum Q value for the next state
        # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        
        """
        Double DQN: compute the maximum Q value for the next state using the Double DQN approach
        $$
        V(s_{t+1}) = Q(s_{t+1}, \arg\max_{a} Q^{\pi}(s_{t+1}, a))
        $$
        """
        # Double DQN: use the policy network to select the action with the maximum Q value for the next state
        # and use the target network to compute the Q value for that action
        # Selection: Get best action indices from policy_net
        next_actions = policy_net(non_final_next_states).max(1).indices.unsqueeze(1)

        # 2. Evaluation: Get Q-values for those actions from target_net
        next_state_values[non_final_mask] = target_net(non_final_next_states).gather(1, next_actions).squeeze(1)

    """
    Compute the expected Q values
    $$
    Q_t(s_t, a_t) = r_t + \gamma \max_{a} Q_{t+1}(s_{t+1}, a)
    $$
    """
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    """
    Compute temporal difference error $\delta=TD(0)$
    
    $$
    \delta_t = Q(s_t, a_t) - (r_t + \gamma \max_{a} Q(s_{t+1}, a))
    $$
    
    Compute Huber loss
    $$
    L(\delta) = \begin{cases} \frac{1}{2} \delta^2 & \text{for } |\delta| < 1 \\ |\delta| - \frac{1}{2} & \text{otherwise} \end{cases}
    $$
    
    $$
    J = \mathbb{E}[L(\delta)] = \frac{1}{|B|} \sum_{i \in B} L(\delta)
    $$
    """
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), clip_value=100)
    
    # update the weights of the policy network
    optimizer.step()


def train():
    print('Training started...')
    for i_episode in range(num_episodes):
        # Initialize the environment and get its state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count(): # Loop until done
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                # next state tensor is the observation after taking the action
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()

            # dictionaries that map each layer's name (the key) to its corresponding parameter torch.Tensor (the weights/biases)
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()
            
            """
            Soft update of the target network's weights: 
            move the target network's weights slightly toward the policy network's weights at every single training step.
            This creates a target that is stationary in the short term but smoothly tracks the policy network in the long term, 
            preventing the "moving target" instability of DQN without the abrupt shocks of a hard update.
            $$
            \theta_{target} = \tau \theta_{policy} + (1 - \tau) \theta_{target}
            $$
            """
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
            
            # update the target network with the new parameters after soft update
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_durations.append(t + 1)
                log_episode_stats(i_episode)
                break


def evaluate_model():
    env_rendered = gym.make("CartPole-v1", render_mode="human")
    # run the trained policy network for 10 episodes, and render the environment
    for i in range(10):
        state, info = env_rendered.reset() # reset environment
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            with torch.no_grad(): # disable gradient calculation for evaluation
                # select the action with the highest Q-value according to the policy network
                action = policy_net(state).max(1).indices.view(1, 1)
            # take the action in the environment
            observation, reward, terminated, truncated, _ = env_rendered.step(action.item())
            # check if episode is done
            done = terminated or truncated
            if done:
                break
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


def main():
    train()
    print('Completed Training')
    evaluate_model()
    print('Completed Evaluation')


if __name__ == "__main__":
    main()
