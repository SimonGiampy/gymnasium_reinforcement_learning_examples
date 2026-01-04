"""
Reinforcement Learning (PPO) with TorchRL Tutorial

This tutorial demonstrates how to use PyTorch and `torchrl` to train a parametric policy
network to solve the tasks from
[OpenAI-Gym/Farama-Gymnasium control library](https://github.com/Farama-Foundation/Gymnasium).

Make sure you have the required packages installed:
```bash
pip3 install torchrl gymnasium[mujoco] dm_control
```

Proximal Policy Optimization (PPO) is a policy-gradient algorithm where a
batch of data is being collected and directly consumed to train the policy to maximise
the expected return given some proximality constraints. For more information, see the
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) paper.

PPO is usually regarded as a fast and efficient method for online, on-policy
reinforcement algorithm. TorchRL provides a loss-module that does all the work
for you, so that you can rely on this implementation and focus on solving your
problem rather than re-inventing the wheel every time you want to train a policy.
"""


# python3 imports

# pytorch imports
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

# torchrl imports
from torchrl.envs import EnvBase  # Base environment wrapper
from torchrl.envs import ParallelEnv  # Parallel environment wrapper
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
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
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

device = torch.device("cpu")


def set_device():
    global device
    # set device
    device = (
        torch.device(0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)


######################################################################
# Hyperparameters
# -------------------

# PPO-clip parameters
MAX_STEPS = 1000  # max steps per episode
SUB_BATCH_SIZE = 1024  # cardinality of the sub-samples gathered from the current data in the inner loop
NUM_EPOCHS = 10  # optimization steps per batch of data collected
CLIP_EPSILON = 0.2  # clip value for PPO loss: see the equation in the intro for more context.
GAMMA = 0.99
LAMBDA = 0.90
ENTROPY_EPS = 1e-2  # entropy bonus coefficient
NORM_EPS = 1e-6  # observation norm scale epsilon

# Data collection parameters

# frames per batch = number of environment steps per data collection batch
FRAMES_PER_BATCH = 1024 * 16

# total frames = total number of environment steps for the whole training
ITERATIONS = 100
TOTAL_FRAMES = FRAMES_PER_BATCH * ITERATIONS

# optimizer parameters
MAX_LR = 5e-4  # starting learning rate for the optimizer
MIN_LR = 1e-5  # minimum learning rate for the scheduler
max_grad_norm = 1.0


def set_max_steps(max_steps: int):
    global MAX_STEPS
    MAX_STEPS = max_steps


def set_frames_iterations(frames_per_batch: int, total_frames: int, iterations: int, sub_batch_size: int):
    global FRAMES_PER_BATCH
    FRAMES_PER_BATCH = frames_per_batch
    global TOTAL_FRAMES
    TOTAL_FRAMES = total_frames
    global ITERATIONS
    ITERATIONS = iterations
    global SUB_BATCH_SIZE
    SUB_BATCH_SIZE = sub_batch_size


def set_lr(max_learning_rate: float, min_learning_rate: float):
    global MAX_LR
    MAX_LR = max_learning_rate
    global MIN_LR
    MIN_LR = min_learning_rate

######################################################################
# Environment definition: use Gymnasium's InvertedDoublePendulum-v5
# defined using the TorchRL GymEnv wrapper for compatibility.


class ObsNormFloat32(nn.Module):
    def __init__(self, loc: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        # input tensors must be allocated, or cloned from the original values
        # allocate class object variables
        # input data must be of dtype float64
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Arithmetic in float64
        self.scale.clamp_min(NORM_EPS)  # avoid division by zero
        obs_norm = (obs - self.loc) / self.scale
        # Cast to float32 for the MLP
        return obs_norm.to(torch.float32)


######################################################################
# Policy
# ------
#

layer_neurons = 256  # number of neurons per linear layer
depth = 4  # number of hidden layers in the MLP policy network


def create_actor_network(env: EnvBase):
    observation_space_num = env.observation_spec["observation"].shape[-1]
    action_space_num = env.action_spec.shape[-1]

    # network is a MLP with three hidden layers and Tanh activation function

    # tanh is a good default choice for continuous control tasks, since it
    # outputs values in the -1, 1 range, and is smooth and non-linear, allowing
    # for smooth policy outputs and good gradient flow

    # output layer has 2 * action_dim outputs, representing the parameters of a
    # Gaussian distribution (mean and standard deviation) for each action dimension

    deep_layers = []
    deep_layers.append(nn.Linear(observation_space_num, layer_neurons, device=device))
    deep_layers.append(nn.Tanh())

    for _ in range(depth):
        deep_layers.append(nn.Linear(layer_neurons, layer_neurons, device=device))
        deep_layers.append(nn.Tanh())

    deep_layers.append(nn.Linear(layer_neurons, 2 * action_space_num, device=device))

    actor_network = nn.Sequential(
        *deep_layers,  # unpack the list of layers
        NormalParamExtractor(),
    )

    return actor_network


def setup_policy(env: EnvBase, actor_network: nn.Module):

    # compute number of parameters in the policy network
    num_policy_params = sum(p.numel() for p in actor_network.parameters())
    print("Number of parameters in the policy network:", num_policy_params)

    # wrap the network in a TensorDictModule to specify input and output keys
    policy_module = TensorDictModule(
        module=actor_network,
        in_keys=["obs_float"],  # observations as input (state)
        out_keys=["loc", "scale"]  # mu, sigma for the TanhNormal distribution
    )

    # create the stochastic actor policy based on the policy network defined above
    probabilistic_actor_policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec,  # action specs from the environment
        in_keys=["loc", "scale"],  # read mu, sigma distribution parameters from these keys
        out_keys=["action"],  # output sampled actions to this key
        distribution_class=TanhNormal,  # apply a TanhNormal distribution for bounded actions
        distribution_kwargs={  # specify action bounds (min, max)
            "low": env.action_spec_unbatched.space.low,
            "high": env.action_spec_unbatched.space.high,
        },
        return_log_prob=True,  # log-probabilities will be computed by the policy
    )

    return probabilistic_actor_policy.to(device)

######################################################################
# Value network
# -------------
#


def setup_value_module(env: EnvBase):

    observation_space_num = env.observation_spec["observation"].shape[-1]
    # define the value network architecture, similar to the policy network
    # output is a single scalar value, representing the estimated value of the input state
    value_network = nn.Sequential(
        nn.Linear(observation_space_num, layer_neurons, device=device),
        nn.Tanh(),
        nn.Linear(layer_neurons, layer_neurons, device=device),
        nn.Tanh(),
        nn.Linear(layer_neurons, layer_neurons, device=device),
        nn.Tanh(),
        nn.Linear(layer_neurons, 1, device=device),
    )

    # compute number of parameters in the value network
    num_value_params = sum(p.numel() for p in value_network.parameters())
    print("Number of parameters in the value network:", num_value_params)

    # wrap the value network in a TensorDictModule
    value_module = TensorDictModule(
        module=value_network,
        in_keys=["obs_float"],
        out_keys=["state_value"],
    )

    return value_module.to(device)


######################################################################
# Data collector
# --------------
#

def create_collector(env: EnvBase, actor_policy: TensorDictModule):

    # data collector to gather data from the environment using the current policy
    # synchronous collector: collects data in the main thread
    collector = SyncDataCollector(
        create_env_fn=env,  # environment to collect data from
        policy=actor_policy,  # current policy to use for data collection
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=TOTAL_FRAMES,
        split_trajs=False,
        device=device,
        storing_device=device,
    )

    return collector


def create_multiprocess_collector(env: EnvBase, actor_policy: TensorDictModule, num_workers: int):

    # data collector to gather data from the environment using the current policy
    # multiprocess collector: collects data in multiple parallel processes
    collector = MultiSyncDataCollector(
        create_env_fn=lambda: env,  # environment to collect data from
        policy=actor_policy,  # current policy to use for data collection
        frames_per_batch=FRAMES_PER_BATCH,
        total_frames=TOTAL_FRAMES,
        split_trajs=False,
        device=device,
        storing_device=device,
        num_workers=num_workers,
    )

    return collector

######################################################################
#
# Replay buffer
# -------------
#


def create_replay_buffer():
    # Replay Buffer Tutorial: https://docs.pytorch.org/rl/stable/tutorials/rb_tutorial.html
    # replay buffer to store collected data for training
    # random sampling without replacement from the collected batch of data
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=FRAMES_PER_BATCH),
        sampler=SamplerWithoutReplacement(),
    )
    return replay_buffer

######################################################################
# Learning modules: advantage estimator, loss function, optimizer, scheduler
# -------------
#


def setup_learning_modules(probabilistic_actor_policy, value_module):

    # Generalized Advantage Estimation (GAE) module to compute advantage values
    advantage_module = GAE(
        gamma=GAMMA, lmbda=LAMBDA,
        value_network=value_module,  # estimator of the value function (approximated by the value network)
        average_gae=True  # average advantages over the batch
    )

    # PPO loss module -- implements the PPO-clip loss function and computes
    # the various loss terms (policy objective, value function loss, entropy bonus)
    # refer to the docs: https://docs.pytorch.org/rl/main/reference/generated/torchrl.objectives.ClipPPOLoss.html
    loss_module = ClipPPOLoss(
        actor_network=probabilistic_actor_policy,  # policy (actor) network
        critic_network=value_module,  # value (critic) network
        clip_epsilon=CLIP_EPSILON,  # clip parameter for PPO loss
        reduction="mean",  # average the loss over the batch
        entropy_bonus=True,  # include an entropy bonus to encourage exploration
        entropy_coeff=ENTROPY_EPS,  # coefficient for the entropy bonus
        critic_coeff=1.0,  # coefficient for the value function loss
        loss_critic_type="smooth_l1",  # Huber loss for value function loss
    )

    optimizer = torch.optim.Adam(
        params=loss_module.parameters(),
        lr=MAX_LR,  # initial learning rate
    )

    # compute the number of batches for the collector
    max_scheduler_steps = TOTAL_FRAMES // FRAMES_PER_BATCH
    print("Total batches of data sampled from the collector:", max_scheduler_steps)
    sub_batch_frames = FRAMES_PER_BATCH // SUB_BATCH_SIZE
    print("Number of sub-batches per batch of data collected:", sub_batch_frames)

    # learning rate scheduler: cosine annealing over the total number of frames
    # the learning rate will decrease from the initial value to 0 over training
    # it follows a half-cosine curve from initial value to 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=max_scheduler_steps,  # number of steps for a full cosine annealing cycle
        eta_min=MIN_LR  # minimum learning rate at the end of training
    )

    return advantage_module, loss_module, optimizer, scheduler

######################################################################
# Logging functions
# -----------------


def log_stats(batch_i: int, tensordict_data, lr):
    train_mean_reward = tensordict_data["next", "reward"].mean().item()
    train_cum_reward = tensordict_data["next", "reward"].sum().item()
    steps = tensordict_data["step_count"].max().item()
    current_lr = lr

    mean_reward_str = f" mean reward={train_mean_reward:.3f}"
    cum_reward_str = f" cum reward={train_cum_reward:.3f}"
    stepcount_str = f" max step-count={steps}"
    lr_str = f" lr={current_lr:.2e}"  # scientific notation

    print(f"epoch {batch_i}:" + mean_reward_str + "," + cum_reward_str + "," + stepcount_str + "," + lr_str)


def log_eval_stats(batch_i: int, eval_rollout):
    eval_reward_mean = eval_rollout["next", "reward"].mean().item()
    eval_reward_sum = eval_rollout["next", "reward"].sum().item()
    steps = eval_rollout["step_count"].max().item()

    mean_reward_str = f" eval mean reward={eval_reward_mean:.3f}"
    cum_reward_str = f" eval cum reward={eval_reward_sum:.3f}"
    stepcount_str = f" eval max step-count={steps}"

    print(f"eval epoch {batch_i}:" + mean_reward_str + "," + cum_reward_str + "," + stepcount_str)

######################################################################
# Training and evaluation functions
# ---------------------------------


def train_sub_batch(replay_buffer, loss_module, optimizer):
    sub_batch_frames = FRAMES_PER_BATCH // SUB_BATCH_SIZE

    # inner loop: iterate over sub-batches sampled from the replay buffer
    for _ in range(sub_batch_frames):
        # sample a sub-batch from the replay buffer
        subdata = replay_buffer.sample(SUB_BATCH_SIZE)

        # compute the PPO loss values for the current sub-batch
        loss_vals = loss_module(subdata.to(device))

        # total loss is the sum of the individual loss terms
        loss_value = (
            loss_vals["loss_objective"]  # clip ppo loss of actor policy network
            + loss_vals["loss_critic"]  # value function loss of critic value network
            + loss_vals["loss_entropy"]  # entropy bonus loss to encourage exploration
        )

        # Optimization: backward, gradient clipping and optimization step
        loss_value.backward()

        # keep gradients from exploding by clipping them to a maximum norm
        torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)

        # perform optimization step: update networks parameters
        optimizer.step()
        optimizer.zero_grad()  # reset gradients for the next step


def eval_policy(batch_i: int, env, probabilistic_actor_policy):
    # evaluation rollout: run the policy in deterministic mode with pure exploitation
    # exploit: take the expected value of the action distribution for a given
    # number of steps
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # execute a rollout with the trained policy for max_steps
        eval_rollout = env.rollout(MAX_STEPS, policy=probabilistic_actor_policy)
        log_eval_stats(batch_i, eval_rollout)
        del eval_rollout


######################################################################
# Exporting and Saving the trained policy for inference
# -----------------------------------------------------

def export_policy(env: EnvBase, actor_policy: TensorDictSequential, model_filepath: str):

    # if the policy improved during evaluation, save the policy network module

    # generate fake tensordict = input data
    fake_td = env.fake_tensordict()
    obs = fake_td["observation"].to(device)

    # remove extension to filepath and add .pt2 for exported module
    if not model_filepath.endswith(".pt2"):
        model_filepath = model_filepath[:-4]
        model_filepath += ".pt2"

    # pure exploitation policy when exporting module program
    # exploitation will take maximum value of mean probabilities
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        exported_policy = torch.export.export(
            actor_policy,
            args=(),
            kwargs={"observation": obs},
            strict=False,
        )

        # print("Deterministic policy graph module: ", exported_policy.graph_module.print_readable())

        output = exported_policy.module()(observation=obs)
        print("Exported module output with fake observation input:", output)

        # save the exported policy module to a file
        torch.export.save(exported_policy, model_filepath)


def save_model_weights(probabilistic_actor_policy, model_filepath: str = "models/model.pth"):
    # save the model weights of the deep MLP model used as policy network
    mlp_model = probabilistic_actor_policy.module[0].module

    # save model weights only with state_dict
    torch.save(mlp_model.state_dict(), model_filepath)
    print("Saved best model weights to file:", model_filepath)


def load_policy_transform(env: EnvBase, transform_module: TensorDictModule,
                          model_filepath: str = "models/model.pth"):

    # load model weights as state_dict from file
    actor_policy_state_dict = torch.load(model_filepath, map_location=device)
    actor_network = create_actor_network(env)

    # load saved model weights into the actor network
    actor_network.load_state_dict(actor_policy_state_dict)

    # setup the probabilistic actor policy from the loaded network
    probabilistic_actor_policy = setup_policy(env, actor_network)

    # create tensordict from normalization and conversion transform
    policy_tensordictseq = TensorDictSequential(
        transform_module,
        # probabilistic actor policy trained to be exported
        probabilistic_actor_policy.requires_grad_(False),
    )
    policy_tensordictseq.select_out_keys("action")
    print("Policy transform input keys:", policy_tensordictseq.in_keys)
    print("Policy transform output keys:", policy_tensordictseq.out_keys)
    return policy_tensordictseq


######################################################################
# Training loop
# -------------


def training_loop(env, probabilistic_actor_policy, value_module, replay_buffer, collector, model_filepath):

    # setup learning modules: advantage estimator, loss function, optimizer, scheduler
    advantage_module, loss_module, optimizer, scheduler = setup_learning_modules(
        probabilistic_actor_policy, value_module)

    # enable gradients for the policy network training
    probabilistic_actor_policy.requires_grad_(True)

    # track the best cumulative reward during evaluation
    best_cum_reward = -float("inf")

    # iteration over the collector allows to get batches of data from the environment
    for i, tensordict_data in enumerate(collector):
        # iterate for a number of epochs for each batch of data collected
        for _ in range(NUM_EPOCHS):

            # compute advantage value using the GAE module estimator
            # the advantage value is recomputed at each epoch since it depends on the policy
            # network which is updated in the inner loop
            advantage_module(tensordict_data)  # computes advantages and adds them to tensordict_data

            # store the collected data in the replay buffer
            data_view = tensordict_data.reshape(-1)
            replay_buffer.extend(data_view.cpu())

            # train on sub-batches sampled from the replay buffer
            train_sub_batch(replay_buffer, loss_module, optimizer)

        current_lr = optimizer.param_groups[0]["lr"]
        log_stats(i, tensordict_data, current_lr)

        # policy evaluation every 5 batches of data collected
        if (i + 1) >= 10 and (i + 1) % 5 == 0:
            # evaluation rollout: run the policy in deterministic mode with pure exploitation
            # exploit: take the expected value of the action distribution for a given
            # number of steps
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                # evaluate the policy multiple times to reduce variance
                for k in range(3):
                    # execute a rollout with the trained policy for max_steps
                    eval_rollout = env.rollout(MAX_STEPS, policy=probabilistic_actor_policy)

                    log_eval_stats(k, eval_rollout)
                    current_cum_reward = eval_rollout["next", "reward"].sum().item()

                    # if the current cumulative reward is better than the best so far, the model has improved
                    # therefore, save the model weights as a checkpoint
                    if best_cum_reward == -float("inf") or int(current_cum_reward) >= int(best_cum_reward):
                        save_model_weights(probabilistic_actor_policy, model_filepath)
                        best_cum_reward = current_cum_reward

                    del eval_rollout  # free memory

        # update the learning rate according to the scheduler after each batch
        scheduler.step()


def run_inference(env: EnvBase, trained_actor_policy: TensorDictModule, episodes: int = 5):

    # set evaluation mode for inference
    # trained_actor_policy.eval()

    # run inference with the trained policy for a number of episodes
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # run multiple episodes of inference with max_steps per episode
        for episode in range(episodes):
            # reset the environment at the start of each episode
            tensordict = env.reset()
            cum_reward = 0.0
            step_count = 0

            # automatic handling of the rollout with the trained policy
            # it takes care of stepping through the environment until done
            inference_rollout = env.rollout(
                MAX_STEPS,
                policy=trained_actor_policy,
                auto_cast_to_device=True,
            )

            # get statistics from the inference rollout
            cum_reward += inference_rollout["next", "reward"].sum().item()
            mean_reward = inference_rollout["next", "reward"].mean().item()
            steps = inference_rollout["next", "reward"].shape[0]

            print(f"Inference Episode {episode + 1}: mean reward = {mean_reward:.3f}, "
                  f"cumulative reward = {cum_reward:.3f}, steps = {steps}")


class ExportedPolicyModule(torch.nn.Module):
    def __init__(self, policy: torch.fx.GraphModule):
        super().__init__()
        self.policy = policy

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # forward method to process observations and return actions
        return self.policy(observation=observation)


def run_inference_exported_model(env: EnvBase, model_filepath: str, episodes=5):

    # load the exported policy module from file
    exported_policy = torch.export.load(model_filepath)
    print("Loaded exported policy module from file:", model_filepath)

    policy = exported_policy.module()
    policy_module = ExportedPolicyModule(policy).to(device)

    run_inference(env, policy_module, episodes)
