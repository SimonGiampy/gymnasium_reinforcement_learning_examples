# torchrl imports
from torchrl.envs import EnvBase, ParallelEnv # environment wrapper
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.utils import check_env_specs

NORM_EPS = 1e-6  # observation norm scale epsilon
MAX_STEPS = 1000  # max steps per episode before termination


def setup_env(base_env: EnvBase):

    # apply a set of transforms to the base environment to:
    # - normalize observations
    # - convert all double tensors to float tensors
    # - add a step counter to keep track of episode lengths before termination
    norm_transform = ObservationNorm(in_keys=["observation"], out_keys=["obs_norm"],
                                     standard_normal=True, eps=NORM_EPS)

    # create the transformed environment via a composition of transforms
    env = TransformedEnv(
        base_env=base_env,
        transform=Compose(  # a sequence of transforms to apply
            norm_transform,  # normalize observations
            DoubleToFloat(in_keys=["obs_norm"], out_keys=["obs_float"]),  # convert all double tensors to float tensors
            StepCounter(max_steps=MAX_STEPS),  # count the steps taken in each episode before termination
        ),
    )

    # initialize mu, sigma for parent environment's observation spec
    norm_transform.init_stats(num_iter=MAX_STEPS,  # number of iterations to run for normalization statistics computation
                              reduce_dim=0, cat_dim=0)  # dimension axis to reduce and concatenate on (data axis)

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

    return env


def setup_base_env(base_env: EnvBase):

    # create the transformed environment via a composition of transforms
    env = TransformedEnv(
        base_env=base_env,
        transform=Compose(  # a sequence of transforms to apply
            StepCounter(max_steps=MAX_STEPS),  # count the steps taken in each episode before termination
        ),
    )

    # sanity check with a dummy rollout
    check_env_specs(env)

    return env


def setup_parallel_env(base_env: EnvBase, num_envs: int):

    parallel_env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=lambda: base_env
    )

    # apply a set of transforms to the base environment to:
    # - normalize observations
    # - convert all double tensors to float tensors
    # - add a step counter to keep track of episode lengths before termination
    norm_transform = ObservationNorm(in_keys=["observation"], out_keys=["obs_norm"],
                                     standard_normal=True, eps=NORM_EPS)

    # create the transformed environment via a composition of transforms
    env = TransformedEnv(
        base_env=parallel_env,
        transform=Compose(  # a sequence of transforms to apply
            norm_transform,  # normalize observations
            DoubleToFloat(in_keys=["obs_norm"], out_keys=["obs_float"]),  # convert all double tensors to float tensors
            StepCounter(max_steps=MAX_STEPS),  # count the steps taken in each episode before termination
        ),
    )

    # initialize mu, sigma for parent environment's observation spec
    norm_transform.init_stats(num_iter=10 * MAX_STEPS,  # number of iterations to run for normalization statistics computation
                              reduce_dim=1, cat_dim=1)  # dimension axis to reduce and concatenate on (tensors axis)

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

    return env
