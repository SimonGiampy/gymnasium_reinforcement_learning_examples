"""
Reinforcement Learning (PPO) with TorchRL Tutorial

**Author**: Vincent Moens https://github.com/vmoens

This tutorial demonstrates how to use PyTorch and `torchrl` to train a parametric policy
network to solve the Inverted Pendulum task from the [OpenAI-Gym/Farama-Gymnasium
control library](https://github.com/Farama-Foundation/Gymnasium).

![Inverted pendulum](https://pytorch.org/tutorials/_static/img/invpendulum.gif)
"""

"""
Make sure you have the required packages installed:
```bash
pip3 install torchrl gymnasium[mujoco] tqdm
"""

"""
Proximal Policy Optimization (PPO) is a policy-gradient algorithm where a
batch of data is being collected and directly consumed to train the policy to maximise
the expected return given some proximality constraints. You can think of it
as a sophisticated version of `REINFORCE <https://link.springer.com/content/pdf/10.1007/BF00992696.pdf>`_,
the foundational policy-optimization algorithm. For more information, see the
`Proximal Policy Optimization Algorithms <https://arxiv.org/abs/1707.06347>`_ paper.

PPO is usually regarded as a fast and efficient method for online, on-policy
reinforcement algorithm. TorchRL provides a loss-module that does all the work
for you, so that you can rely on this implementation and focus on solving your
problem rather than re-inventing the wheel every time you want to train a policy.
"""


# python3 imports
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm

# pytorch imports
import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

# torchrl imports
from torchrl.envs.libs.gym import GymEnv # Gymnasium environment wrapper
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# set device
device = (
    torch.device(0)
    if torch.cuda.is_available()
    else torch.device("cpu")
)
print("Using device:", device)

######################################################################
# PPO-clip parameters

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimization steps per batch of data collected
CLIP_EPSILON = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_EPS = 1e-4

######################################################################
# Environment definition: use Gymnasium's InvertedDoublePendulum-v4
# defined using the TorchRL GymEnv wrapper for compatibility.

print("list of available Gymnasium environments:", GymEnv.available_envs)

# initialize the base environment
base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

# apply a set of transforms to the base environment to:
# - normalize observations
# - convert all double tensors to float tensors
# - add a step counter to keep track of episode lengths before termination
norm_transform = ObservationNorm(in_keys=["observation"])

# create the transformed environment via a composition of transforms
env = TransformedEnv(
    base_env=base_env,
    transform=Compose( # a sequence of transforms to apply
        norm_transform, # normalize observations
        DoubleToFloat(), # convert all double tensors to float tensors
        StepCounter(), # count the steps taken in each episode before termination
    ),
)

# initialize mu, sigma for parent environment's observation spec
norm_transform.init_stats(num_iter=1000, # number of iterations to run for normalization statistics computation
                          reduce_dim=0, cat_dim=0) # dimension axis to reduce and concatenate on (batch axis)


print("normalization constant shape:", norm_transform.loc.shape)
print("observation_spec:", env.observation_spec["observation"])
print("reward_spec:", env.reward_spec)
print("action_spec:", env.action_spec)
observation_space_shape = env.observation_spec["observation"].shape
print("size of observation space:", observation_space_shape)
action_space_shape = env.action_spec.shape
print("size of action space:", action_space_shape)

# sanity check with a dummy rollout
check_env_specs(env)

# test a rollout of three steps, and check the shape of the resulting TensorDict
rollout = env.rollout(3)
# print("rollout of three steps:", rollout)
print("Shape of the rollout TensorDict:", rollout.batch_size)

######################################################################
#
# Policy
# ------
#

layer_neurons = 256  # number of neurons per linear layer

# network is a MLP with three hidden layers and Tanh activation function

# tanh is a good default choice for continuous control tasks, since it
# outputs values in the -1, 1 range, and is smooth and non-linear, allowing
# for smooth policy outputs and good gradient flow

# output layer has 2 * action_dim outputs, representing the parameters of a
# Gaussian distribution (mean and standard deviation) for each action dimension

actor_network = nn.Sequential(
    nn.Linear(observation_space_shape[-1], layer_neurons, device=device),
    nn.Tanh(),
    nn.Linear(layer_neurons, layer_neurons, device=device),
    nn.Tanh(),
    nn.Linear(layer_neurons, layer_neurons, device=device),
    nn.Tanh(),
    nn.Linear(layer_neurons, 2 * action_space_shape[-1], device=device),
    NormalParamExtractor(),
)

# compute number of parameters in the policy network
num_policy_params = sum(p.numel() for p in actor_network.parameters())
print("Number of parameters in the policy network:", num_policy_params)

# wrap the network in a TensorDictModule to specify input and output keys
policy_module = TensorDictModule(
    module=actor_network,
    in_keys=["observation"], # observations as input (state)
    out_keys=["loc", "scale"] # mu, sigma for the TanhNormal distribution
)

# create the stochastic actor policy based on the policy network defined above
probabilistic_actor_policy = ProbabilisticActor(
    module=policy_module,
    spec=env.action_spec, # action specs from the environment
    in_keys=["loc", "scale"], # read mu, sigma distribution parameters from these keys
    distribution_class=TanhNormal, # apply a TanhNormal distribution for bounded actions
    distribution_kwargs={ # specify action bounds (min, max)
        "low": env.action_spec_unbatched.space.low,
        "high": env.action_spec_unbatched.space.high,
    },
    return_log_prob=True, # log-probabilities will be computed by the policy
)

######################################################################
#
# Value network
# -------------
#

# define the value network architecture, similar to the policy network
# output is a single scalar value, representing the estimated value of the input state
value_network = nn.Sequential(
    nn.Linear(observation_space_shape[-1], layer_neurons, device=device),
    nn.Tanh(),
    nn.Linear(layer_neurons, layer_neurons, device=device),
    nn.Tanh(),
    nn.Linear(layer_neurons, layer_neurons, device=device),
    nn.Tanh(),
    nn.Linear(layer_neurons, 1, device=device),
)

# wrap the value network in a TensorDictModule
value_module = TensorDictModule(
    module=value_network,
    in_keys=["observation"],
    out_keys=["state_value"],
)

print("Running policy:", probabilistic_actor_policy(env.reset()))
print("Running value:", value_module(env.reset()))

######################################################################
# 
# Data collector
# --------------
#

# Data collection parameters

# frames per batch = number of environment steps per data collection batch
frames_per_batch = 1000

# total frames = total number of environment steps for the whole training
total_frames = 10_000

collector = SyncDataCollector(
    create_env_fn=env,
    policy=probabilistic_actor_policy,
    frames_per_batch=frames_per_batch,
    total_frames=total_frames,
    split_trajs=False,
    device=device,
)

######################################################################
#
# Replay buffer
# -------------
#

# Replay Buffer Tutorial: https://docs.pytorch.org/rl/stable/tutorials/rb_tutorial.html
# replay buffer to store collected data for training
# random sampling without replacement from the collected batch of data
replay_buffer = ReplayBuffer(
    storage=LazyTensorStorage(max_size=frames_per_batch),
    sampler=SamplerWithoutReplacement(),
)

######################################################################
#
# Loss function
# -------------
#

# Generalized Advantage Estimation (GAE) module to compute advantage values
advantage_module = GAE(
    gamma=GAMMA, lmbda=LAMBDA, 
    value_network=value_module, # estimator of the value function (approximated by the value network)
    average_gae=True # average advantages over the batch
)

# PPO loss module -- implements the PPO-clip loss function and computes
# the various loss terms (policy objective, value function loss, entropy bonus)
loss_module = ClipPPOLoss(
    actor_network=probabilistic_actor_policy, # policy (actor) network
    critic_network=value_module, # value (critic) network
    clip_epsilon=CLIP_EPSILON, # clip parameter for PPO loss
    entropy_bonus=bool(ENTROPY_EPS), 
    entropy_coeff=ENTROPY_EPS,
    # these keys match by default but we set this for completeness
    critic_coeff=1.0,
    loss_critic_type="smooth_l1", # Huber loss for value function loss
)

lr = 3e-4
max_grad_norm = 1.0
optim = torch.optim.Adam(
    params=loss_module.parameters(), 
    lr=lr
)

# learning rate scheduler: cosine annealing over the total number of frames
# the learning rate will decrease from the initial value to 0 over training
# it follows a half-cosine curve from initial value to 0

# compute the number of scheduler update steps over the whole training
max_scheduler_steps = total_frames // frames_per_batch
print("Total scheduler steps:", max_scheduler_steps)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optim, 
    T_max=max_scheduler_steps, # number of steps for a full cosine annealing cycle
    eta_min=0.0 # minimum learning rate at the end of training
)

######################################################################
#
# Training loop
# -------------

logs = defaultdict(list)
pbar = tqdm(total=total_frames)
eval_str = ""

# We iterate over the collector until it reaches the total number of frames it was
# designed to collect:
for i, tensordict_data in enumerate(collector):
    # we now have a batch of data to work with. Let's learn something from it.
    for _ in range(num_epochs):
        # We'll need an "advantage" signal to make PPO work.
        # We re-compute it at each epoch as its value depends on the value
        # network which is updated in the inner loop.
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(frames_per_batch // sub_batch_size):
            subdata = replay_buffer.sample(sub_batch_size)
            loss_vals = loss_module(subdata.to(device))
            loss_value = (
                loss_vals["loss_objective"]
                + loss_vals["loss_critic"]
                + loss_vals["loss_entropy"]
            )

            # Optimization: backward, grad clipping and optimization step
            loss_value.backward()
            # this is not strictly mandatory but it's good practice to keep
            # your gradient norm bounded
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
            optim.step()
            optim.zero_grad()

    logs["reward"].append(tensordict_data["next", "reward"].mean().item())
    pbar.update(tensordict_data.numel())
    cum_reward_str = (
        f"average reward={logs['reward'][-1]: 4.4f} (init={logs['reward'][0]: 4.4f})"
    )
    logs["step_count"].append(tensordict_data["step_count"].max().item())
    stepcount_str = f"step count (max): {logs['step_count'][-1]}"
    logs["lr"].append(optim.param_groups[0]["lr"])
    lr_str = f"lr policy: {logs['lr'][-1]: 4.4f}"
    if i % 10 == 0:
        # We evaluate the policy once every 10 batches of data.
        # Evaluation is rather simple: execute the policy without exploration
        # (take the expected value of the action distribution) for a given
        # number of steps (1000, which is our ``env`` horizon).
        # The ``rollout`` method of the ``env`` can take a policy as argument:
        # it will then execute this policy at each step.
        with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
            # execute a rollout with the trained policy
            eval_rollout = env.rollout(1000, policy=probabilistic_actor_policy)
            logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
            logs["eval reward (sum)"].append(
                eval_rollout["next", "reward"].sum().item()
            )
            logs["eval step_count"].append(eval_rollout["step_count"].max().item())
            eval_str = (
                f"eval cumulative reward: {logs['eval reward (sum)'][-1]: 4.4f} "
                f"(init: {logs['eval reward (sum)'][0]: 4.4f}), "
                f"eval step-count: {logs['eval step_count'][-1]}"
            )
            del eval_rollout
    pbar.set_description(", ".join([eval_str, cum_reward_str, stepcount_str, lr_str]))

    # We're also using a learning rate scheduler. Like the gradient clipping,
    # this is a nice-to-have but nothing necessary for PPO to work.
    scheduler.step()

######################################################################
#
# Results
# -------

plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.plot(logs["reward"])
plt.title("training rewards (average)")
plt.subplot(2, 2, 2)
plt.plot(logs["step_count"])
plt.title("Max step count (training)")
plt.subplot(2, 2, 3)
plt.plot(logs["eval reward (sum)"])
plt.title("Return (test)")
plt.subplot(2, 2, 4)
plt.plot(logs["eval step_count"])
plt.title("Max step count (test)")
plt.show()
