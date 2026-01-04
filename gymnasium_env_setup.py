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
from tensordict.nn import TensorDictModule, TensorDictSequential

# local imports
from mujoco_ppo import ObsNormFloat32, load_policy_transform

NORM_EPS = 1e-6  # observation norm scale epsilon
MAX_STEPS = 1000  # max steps per episode before termination


def setup_env_training(base_env: EnvBase):

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


def setup_env_inference(base_env: EnvBase):

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


def setup_inference_transforms(env: EnvBase):
    # transform the policy to include the environment's preprocessing transforms
    # this allows to export a single module that takes raw observations as input
    # and outputs actions directly, without needing to apply the transforms separately

    # get normalization constants from the environment's observation normalization transform
    norm_loc, norm_scale = env.transform[0].loc.clone(), env.transform[0].scale.clone()
    # if loc and scale have shape [num_envs, obs_dim], extract normalization constants of batch 0
    if len(norm_loc.shape) > 1:
        norm_loc = norm_loc[0]
        norm_scale = norm_scale[0]
    # print("Loaded observation normalization: loc = ", norm_loc, "; scale = ", norm_scale)
    
    # create a new observation normalization transform using the constants
    norm_float32_module = TensorDictModule(
        module=ObsNormFloat32(loc=norm_loc, scale=norm_scale), 
        in_keys=["observation"],
        out_keys=["obs_float"]
    )

    return norm_float32_module