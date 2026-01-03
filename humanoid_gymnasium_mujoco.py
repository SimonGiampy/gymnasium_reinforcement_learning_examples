# python 3 imports
import sys

# import local script with all functions
import mujoco_ppo as mj_ppo

# import Gymnasium environment wrapper
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.libs.gym import set_gym_backend

# Humanoid environment:
# https://gymnasium.farama.org/environments/mujoco/humanoid/
env_name = "Humanoid-v5"

num_envs = 32  # number of parallel environments for training


def training():
    mj_ppo.set_device()

    print("list of available Gymnasium environments:", GymEnv.available_envs)

    # initialize environment for training
    with set_gym_backend("gymnasium"):
        base_env = GymEnv(env_name, device=mj_ppo.device, render_mode=None,
                          forward_reward_weight=4.0, # default = 1.25
                          healthy_reward=1.5, # default = 5.0
                          terminate_when_unhealthy=True)
        print("Initialized base environment:", env_name)

    mj_ppo.set_max_steps(1000)
    iterations = 10
    sub_batch_size = 2048
    frames_per_batch = sub_batch_size * 8
    total_frames = frames_per_batch * iterations  # total frames for training
    mj_ppo.set_frames_iterations(
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        iterations=iterations,
        sub_batch_size=sub_batch_size
    )
    mj_ppo.set_lr(min_learning_rate=1e-5, max_learning_rate=5e-4)

    parallel_env = mj_ppo.setup_parallel_env(base_env, num_envs)
    print(f"Initialized parallel environment with {num_envs} workers.")

    # get normalization constants from the environment's observation normalization transform
    print("Parallel Env with transforms:", parallel_env.transform)
    norm_loc, norm_scale = parallel_env.transform[0].loc.clone(), parallel_env.transform[0].scale.clone()
    # extract normalization constants of batch 0
    norm_loc = norm_loc[0]
    norm_scale = norm_scale[0]
    # print("Loaded observation normalization: loc = ", norm_loc, "; scale = ", norm_scale)

    # create policy actor network and value critic network
    actor_network = mj_ppo.create_actor_network(parallel_env)
    probabilistic_actor_policy = mj_ppo.setup_policy(parallel_env, actor_network)
    value_module = mj_ppo.setup_value_module(parallel_env)

    # print("Running policy actor module:", probabilistic_actor_policy(parallel_env.reset()))
    # print("Running value critic module:", value_module(parallel_env.reset()))

    # setup of replay buffer, collector
    collector = mj_ppo.create_collector(parallel_env, probabilistic_actor_policy)
    replay_buffer = mj_ppo.create_replay_buffer()

    # setup learning modules, then run the training loop
    mj_ppo.training_loop(parallel_env, probabilistic_actor_policy, value_module,
                         replay_buffer, collector, model_filepath="models/humanoid_gymnasium_ppo.pth")
    
    # initialize the base environment with rendering enabled
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode="human")
    print("Initialized rendered environment:", env_name)

    # create environment with rendering enabled for inference
    render_env = mj_ppo.setup_env(base_env)

    # load best policy from saved model weights
    trained_actor_policy = mj_ppo.load_policy_norm(env=render_env, norm_loc=norm_loc, norm_scale=norm_scale,
                                                   model_filepath="models/humanoid_gymnasium_ppo.pth")
    
    # save model program
    mj_ppo.export_policy(env=render_env, actor_policy=trained_actor_policy,
                         model_filepath="models/humanoid_gymnasium_ppo.pt2")
    
    # run inference with the trained policy for a number of episodes
    mj_ppo.run_inference(render_env, trained_actor_policy, episodes=10)

    # parallel_env.close()
    render_env.close()

def inference():
    mj_ppo.set_device()
    
    # initialize the base environment with rendering enabled
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode="human")
    print("Initialized rendered environment:", env_name)
    
    saved_model_path = "models/humanoid_gymnasium_ppo.pt2"
    
    # run inference with the trained policy for a number of episodes
    mj_ppo.run_inference_exported_model(base_env, saved_model_path, episodes=10)


def main():
    # if run with --train argument, perform training; otherwise, run inference
    if "--train" in sys.argv:
        training()
    elif "--inference" in sys.argv:
        inference()
    else:
        print("Please specify either --train or --inference argument to run the desired mode.")
    
if __name__ == "__main__":
    main()
