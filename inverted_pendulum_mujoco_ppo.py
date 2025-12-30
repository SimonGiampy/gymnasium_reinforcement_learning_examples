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

# pytorch imports
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

# torchrl imports
from torchrl.envs.libs.gym import GymEnv  # Gymnasium environment wrapper
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
from torchrl.modules import ProbabilisticActor, TanhNormal
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
max_steps = 1000  # max steps per episode
sub_batch_size = 128  # cardinality of the sub-samples gathered from the current data in the inner loop
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


def init_env():

    print("list of available Gymnasium environments:", GymEnv.available_envs)

    # initialize the base environment
    base_env = GymEnv("InvertedDoublePendulum-v4", device=device)

    # apply a set of transforms to the base environment to:
    # - normalize observations
    # - convert all double tensors to float tensors
    # - add a step counter to keep track of episode lengths before termination
    norm_transform = ObservationNorm(in_keys=["observation"], out_keys=["obs_norm"])

    # create the transformed environment via a composition of transforms
    env = TransformedEnv(
        base_env=base_env,
        transform=Compose(  # a sequence of transforms to apply
            norm_transform,  # normalize observations
            DoubleToFloat(in_keys=["obs_norm"], out_keys=["obs_float"]),  # convert all double tensors to float tensors
            StepCounter(max_steps=max_steps),  # count the steps taken in each episode before termination
        ),
    )

    # initialize mu, sigma for parent environment's observation spec
    norm_transform.init_stats(num_iter=max_steps,  # number of iterations to run for normalization statistics computation
                              reduce_dim=0, cat_dim=0)  # dimension axis to reduce and concatenate on (batch axis)

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

    return env

######################################################################
# Policy
# ------
#


layer_neurons = 256  # number of neurons per linear layer


def setup_policy(env):
    observation_space_num = env.observation_spec["observation"].shape[-1]
    action_space_num = env.action_spec.shape[-1]

    # network is a MLP with three hidden layers and Tanh activation function

    # tanh is a good default choice for continuous control tasks, since it
    # outputs values in the -1, 1 range, and is smooth and non-linear, allowing
    # for smooth policy outputs and good gradient flow

    # output layer has 2 * action_dim outputs, representing the parameters of a
    # Gaussian distribution (mean and standard deviation) for each action dimension

    actor_network = nn.Sequential(
        nn.Linear(observation_space_num, layer_neurons, device=device),
        nn.Tanh(),
        nn.Linear(layer_neurons, layer_neurons, device=device),
        nn.Tanh(),
        nn.Linear(layer_neurons, layer_neurons, device=device),
        nn.Tanh(),
        nn.Linear(layer_neurons, 2 * action_space_num, device=device),
        NormalParamExtractor(),
    )

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

    return probabilistic_actor_policy

######################################################################
# Value network
# -------------
#


def setup_value_module(env):

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

    return value_module


######################################################################
# Data collector
# --------------
#

# Data collection parameters

# frames per batch = number of environment steps per data collection batch
frames_per_batch = 1024

# total frames = total number of environment steps for the whole training
total_frames = 1024 * 10


def create_collector(env: GymEnv, actor_policy):

    # data collector to gather data from the environment using the current policy
    # synchronous collector: collects data in the main thread
    collector = SyncDataCollector(
        create_env_fn=env,  # environment to collect data from
        policy=actor_policy,  # current policy to use for data collection
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device,
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
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    return replay_buffer

######################################################################
# Learning modules: advantage estimator, loss function, optimizer, scheduler
# -------------
#


lr = 1e-3  # starting learning rate for the optimizer
max_grad_norm = 1.0

# compute the number of batches for the collector
max_scheduler_steps = total_frames // frames_per_batch
print("Total batches of data sampled from the collector:", max_scheduler_steps)

sub_batch_frames = frames_per_batch // sub_batch_size
print("Number of sub-batches per batch of data collected:", sub_batch_frames)


def setup_learning_modules(env, probabilistic_actor_policy, value_module):

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
        lr=lr
    )

    # learning rate scheduler: cosine annealing over the total number of frames
    # the learning rate will decrease from the initial value to 0 over training
    # it follows a half-cosine curve from initial value to 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=max_scheduler_steps,  # number of steps for a full cosine annealing cycle
        eta_min=5e-5  # minimum learning rate at the end of training
    )

    return advantage_module, loss_module, optimizer, scheduler

######################################################################
# Logging functions
# -----------------


def log_stats(batch_i, tensordict_data, lr):
    train_reward = tensordict_data["next", "reward"].mean().item()
    steps = tensordict_data["step_count"].max().item()
    current_lr = lr

    cum_reward_str = f" average reward={train_reward: 4.4f}"
    stepcount_str = f" max step-count={steps}"
    lr_str = f" lr={current_lr: 4.4f}"

    print(f"epoch {batch_i}:" + cum_reward_str + "," + stepcount_str + "," + lr_str)


def log_eval_stats(batch_i, eval_rollout):
    eval_reward_mean = eval_rollout["next", "reward"].mean().item()
    eval_reward_sum = eval_rollout["next", "reward"].sum().item()
    steps = eval_rollout["step_count"].max().item()

    mean_reward_str = f" eval mean reward={eval_reward_mean: 4.4f}"
    cum_reward_str = f" eval cum reward={eval_reward_sum: 4.4f}"
    stepcount_str = f" eval max step-count={steps}"

    print(f"eval epoch {batch_i}:" + mean_reward_str + "," + cum_reward_str + "," + stepcount_str)

######################################################################
# Training and evaluation functions
# ---------------------------------


def train_sub_batch(replay_buffer, loss_module, optimizer):
    # inner loop: iterate over sub-batches sampled from the replay buffer
    for _ in range(sub_batch_frames):
        # sample a sub-batch from the replay buffer
        subdata = replay_buffer.sample(sub_batch_size)

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


def eval_policy(batch_i, env, probabilistic_actor_policy):
    # evaluation rollout: run the policy in deterministic mode with pure exploitation
    # exploit: take the expected value of the action distribution for a given
    # number of steps
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        # execute a rollout with the trained policy for max_steps
        eval_rollout = env.rollout(max_steps, policy=probabilistic_actor_policy)
        log_eval_stats(batch_i, eval_rollout)
        del eval_rollout


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
        obs_norm = (obs - self.loc) / (self.scale + 1e-8)
        # Cast to float32 for the MLP
        return obs_norm.to(torch.float32)

######################################################################
# Exporting and Saving the trained policy for inference
# -----------------------------------------------------


def export_policy(env, probabilistic_actor_policy):
    # transform the policy to include the environment's preprocessing transforms
    # this allows to export a single module that takes raw observations as input
    # and outputs actions directly, without needing to apply the transforms separately

    # get normalization constants from the environment's observation normalization transform
    norm_loc, norm_scale = env.transform[0].loc.clone(), env.transform[0].scale.clone()
    print("Normalization loc:", norm_loc)
    print("Normalization scale:", norm_scale)

    # create a new observation normalization transform using the constants
    norm_float32_module = TensorDictModule(
        module=ObsNormFloat32(loc=norm_loc, scale=norm_scale),
        in_keys=["observation"],
        out_keys=["obs_float"]
    )

    # create tensordict from normalization and conversion trasform
    policy_transform = TensorDictSequential(
        norm_float32_module,
        # probabilistic actor policy trained to be exported
        probabilistic_actor_policy.requires_grad_(False),
    )
    policy_transform.select_out_keys("action")
    print("Policy transform input keys:", policy_transform.in_keys)
    print("Policy transform output keys:", policy_transform.out_keys)

    # if the policy improved during evaluation, save the policy network module

    # generate fake tensordict = input data
    fake_td = env.base_env.fake_tensordict()
    obs = fake_td["observation"].to(device)

    # pure exploitation policy when exporting module program
    # exploitation will take maximum value of mean probabilities
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        exported_policy = torch.export.export(
            policy_transform,
            args=(),
            kwargs={"observation": obs},
            strict=False,
        )

        print("Deterministic policy graph module: ", exported_policy.graph_module.print_readable())

        output = exported_policy.module()(observation=obs)
        print("Exported module output with fake observation input:", output)

        # save the exported policy module to a file
        torch.export.save(exported_policy, "inverted_pendulum_ppo_policy.pt2")


def save_model_weights(probabilistic_actor_policy):
    # save model weights only with state_dict
    torch.save(probabilistic_actor_policy.state_dict(), "models/inverted_pendulum_ppo.pth")

######################################################################
# Training loop
# -------------


def training_loop(env, probabilistic_actor_policy, value_module, replay_buffer, collector):

    # setup learning modules: advantage estimator, loss function, optimizer, scheduler
    advantage_module, loss_module, optimizer, scheduler = setup_learning_modules(
        env, probabilistic_actor_policy, value_module)

    probabilistic_actor_policy.requires_grad_(True)
    # iteration over the collector allows to get batches of data from the environment
    for i, tensordict_data in enumerate(collector):
        # iterate for a number of epochs for each batch of data collected
        for _ in range(num_epochs):

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

        # policy evaluation every 10 batches of data collected
        if i >= 10 and i % 10 == 0:
            eval_policy(i, env, probabilistic_actor_policy)

        # update the learning rate according to the scheduler after each batch
        scheduler.step()


def main():

    env = init_env()
    
    # create policy actor network and value critic network
    probabilistic_actor_policy = setup_policy(env)
    value_module = setup_value_module(env)

    # print("Running policy actor module:", probabilistic_actor_policy(env.reset()))
    # print("Running value critic module:", value_module(env.reset()))

    # setup of replay buffer, collector
    collector = create_collector(env, probabilistic_actor_policy)
    replay_buffer = create_replay_buffer()

    # create data collector
    collector = create_collector(env, probabilistic_actor_policy)

    # create replay buffer
    replay_buffer = create_replay_buffer()

    # setup learning modules, then run the training loop
    training_loop(env, probabilistic_actor_policy, value_module, replay_buffer, collector)

    # final evaluation after training is complete
    eval_policy(-1, env, probabilistic_actor_policy)

    # save model program
    export_policy(env, probabilistic_actor_policy)


if __name__ == "__main__":
    main()
