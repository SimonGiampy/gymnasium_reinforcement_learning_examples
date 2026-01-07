# torchrl imports
from torchrl.envs import EnvBase, ParallelEnv  # environment wrapper
from torchrl.envs import (
    Compose,
    DoubleToFloat,
    ObservationNorm,
    StepCounter,
    CatTensors,
    TransformedEnv,
)
from torchrl.envs.utils import check_env_specs
from tensordict.nn import TensorDictModule, TensorDictSequential

# torch imports
import torch

# python3 imports
import numpy as np

# local imports
from mujoco_ppo import ObsNormFloat32

# DeepMind Control library
from dm_control import viewer
from dm_control.rl.control import Environment
from dm_env import TimeStep

NORM_EPS = 1e-6  # observation norm scale epsilon
MAX_STEPS = 10  # max steps per episode before termination (expressed in seconds)
default_max_steps = 25 # default max steps for inference rendering (in seconds)

def setup_env_training(base_env: EnvBase, num_envs: int = 1):
    
    # get list of observation keys from the base environment
    obs_keys = list(base_env.observation_spec.keys())

    # create parallel environment with num_envs workers
    parallel_env = ParallelEnv(
        num_workers=num_envs,
        create_env_fn=lambda: base_env
    )

    # concatenate observation components into single observation tensor
    cat_tensors_module = CatTensors(
        in_keys=obs_keys,
        out_key="observation")

    # apply a set of transforms to the base environment to:
    # - normalize observations
    # - convert all double tensors to float tensors
    # - add a step counter to keep track of episode lengths before termination
    norm_transform = ObservationNorm(in_keys=["observation"], out_keys=["obs_norm"],
                                     standard_normal=True, eps=NORM_EPS)

    env = TransformedEnv(
        base_env=parallel_env,
        transform=Compose(  # a sequence of transforms to apply
            cat_tensors_module,
            norm_transform,
            DoubleToFloat(in_keys=["obs_norm"], out_keys=["obs_float"]),  # convert all double tensors to float tensors
            StepCounter(max_steps=MAX_STEPS),  # count the steps taken in each episode before termination
        )
    )

    batch_dim = 0 if num_envs == 1 else 1  # dimension axis to reduce and concatenate on (tensors axis)

    # initialize mu, sigma for parent environment's observation spec
    norm_transform.init_stats(num_iter=5 * MAX_STEPS,  # number of iterations to run for normalization statistics computation
                              reduce_dim=batch_dim, cat_dim=batch_dim)

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
    
    obs_keys = list(base_env.observation_spec.keys())

    cat_tensors_module = CatTensors(
        in_keys=obs_keys,
        out_key="observation"
    )

    env = TransformedEnv(
        base_env=base_env,
        transform=Compose(  # a sequence of transforms to apply
            cat_tensors_module,
        )
    )

    return env


def setup_inference_transforms(env: EnvBase):
    # transform the policy to include the environment's preprocessing transforms
    # this allows to export a single module that takes raw observations as input
    # and outputs actions directly, without needing to apply the transforms separately

    # get normalization constants from the environment's observation normalization transform
    norm_loc, norm_scale = env.transform[1].loc.clone(), env.transform[1].scale.clone()
    # if loc and scale have shape [num_envs, obs_dim], extract normalization constants of batch 0
    if len(norm_loc.shape) > 1:
        norm_loc = norm_loc[0]
        norm_scale = norm_scale[0]
    
    norm_transform_module = ObsNormFloat32(loc=norm_loc, scale=norm_scale)

    # create a new observation normalization transform using the constants
    transform_module = TensorDictModule(
        module=Compose(
            norm_transform_module,
        ),
        in_keys=["observation"],
        out_keys=["obs_float"]
    )

    return transform_module


def run_inference_rendered(env, policy: TensorDictSequential):
    """
    Args:
        policy: An optional callable corresponding to a policy to execute
            within the environment. It should accept a `TimeStep` and return
            a numpy array of actions conforming to the output of
            `environment.action_spec()`. If the callable implements a method `reset`
            then this method is called when the viewer is reset.
    """
    dm_env = env._env  # get the underlying dm_control environment
    time_step = dm_env.reset()
    device = env.device

    policy.to(device=device)

    def execute_policy(step: TimeStep) -> np.ndarray:
        """
        Execute the policy on a given TimeStep to produce an action.
        Args:
            step: A `TimeStep` namedtuple containing the current
                observation, reward, discount, and step type.
        Returns:
            A numpy array of actions conforming to the output of
            `environment.action_spec()`.
        """
        obs_dict = step.observation  # ordered dict, one key per observation component
        # get numpy arrays from ordered dict
        obs_tensors = {}
        for k, v in obs_dict.items():
            # if v is float, convert to single-element numpy array
            if not isinstance(v, np.ndarray):
                v = np.array([v], dtype=np.float64)
            else:
                v = np.array(v, dtype=np.float64)

            # convert to torch tensor
            obs_tensors[k] = torch.from_numpy(v)

        # concatenate tensors along last dimension to form single observation tensor
        obs_cat = torch.cat(list(obs_tensors.values()), dim=-1)

        # cast to device, and perform inference with the trained policy
        action = policy(obs_cat.to(device))

        # execute the computed action in the environment and display its rendered output
        return action.cpu().numpy()

    # launch the viewer with the environment and the policy to execute
    viewer.launch(dm_env, policy=execute_policy)
