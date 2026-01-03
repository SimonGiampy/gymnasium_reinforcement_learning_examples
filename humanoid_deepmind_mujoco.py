# python3 imports
import sys

# import local script with all functions
import mujoco_ppo as mj_ppo
from deepmind_env_setup import setup_env
import deepmind_env_setup

# import Gymnasium environment wrapper
from torchrl.envs.libs.dm_control import DMControlEnv

# DeepMind Control
from dm_control import viewer

# Humanoid environment:
# https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/humanoid.py
env_name = "humanoid"
task_name = "walk"

num_envs = 2  # number of parallel environments for training

model_file_name = "models/humanoid_deepmind_ppo"

def training():

    print("list of available DeepMind Control environments:", DMControlEnv.available_envs)
    
    mj_ppo.set_device()

    # initialize environment for training
    base_env = DMControlEnv(env_name=env_name, task_name=task_name, 
                            device=mj_ppo.device)
    print("Initialized base environment: ", env_name, ":", task_name)
    #viewer.launch(base_env._env)

    # setting training hyperparameters
    mj_ppo.set_max_steps(deepmind_env_setup.MAX_STEPS)
    
    iterations = 10
    sub_batch_size = 256
    frames_per_batch = sub_batch_size * 8
    total_frames = frames_per_batch * iterations  # total frames for training
    mj_ppo.set_frames_iterations(
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        iterations=iterations,
        sub_batch_size=sub_batch_size
    )
    
    # learning rate scheduler parameters
    mj_ppo.set_lr(min_learning_rate=1e-5, max_learning_rate=5e-4)

    env = setup_env(base_env, num_envs)
    print(f"Initialized parallel environment with {num_envs} workers.")

    # get normalization constants from the environment's observation normalization transform
    print("Parallel Env with transforms:", env.transform)
    norm_loc, norm_scale = env.transform[1].loc.clone(), env.transform[1].scale.clone()
    # extract normalization constants of batch 0
    #norm_loc = norm_loc[0]
    #norm_scale = norm_scale[0]
    print("Loaded observation normalization: loc = ", norm_loc, "; scale = ", norm_scale)

    # create policy actor network and value critic network
    actor_network = mj_ppo.create_actor_network(env)
    probabilistic_actor_policy = mj_ppo.setup_policy(env, actor_network)
    value_module = mj_ppo.setup_value_module(env)

    # print("Running policy actor module:", probabilistic_actor_policy(env.reset()))
    # print("Running value critic module:", value_module(env.reset()))

    # setup of replay buffer, collector
    collector = mj_ppo.create_collector(env, probabilistic_actor_policy)
    replay_buffer = mj_ppo.create_replay_buffer()

    # setup learning modules, then run the training loop
    mj_ppo.training_loop(env, probabilistic_actor_policy, value_module,
                         replay_buffer, collector, model_filepath=model_file_name + ".pth")
    
    # load best policy from saved model weights
    trained_actor_policy = mj_ppo.load_policy_norm(env=base_env, norm_loc=norm_loc, norm_scale=norm_scale,
                                                   model_filepath=model_file_name + ".pth")
    
    # save model program
    mj_ppo.export_policy(env=base_env, actor_policy=trained_actor_policy,
                         model_filepath=model_file_name + ".pt2")

    # run inference with the trained policy for a number of episodes
    mj_ppo.run_inference(base_env, trained_actor_policy, episodes=10)

    # parallel_env.close()
    env.close()

def inference():
    mj_ppo.set_device()
    
    # initialize the base environment with rendering enabled
    base_env = DMControlEnv(env_name=env_name, task_name=task_name,
                            device=mj_ppo.device)
    print("Initialized rendered environment:", env_name)
    
    saved_model_path = model_file_name + ".pt2"
    
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
