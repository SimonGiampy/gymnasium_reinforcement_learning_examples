import sys

# import local script with all functions
from .mujoco_rl import mujoco_ppo as mj_ppo

# import Gymnasium environment wrapper
from torchrl.envs.libs.gym import GymEnv


# Inverted Double Pendulum environment:
# https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/
env_name = "InvertedDoublePendulum-v5"


def training():
    
    mj_ppo.set_device()

    print("list of available Gymnasium environments:", GymEnv.available_envs)

    # initialize environment for training
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode=None)
    print("Initialized base environment:", env_name)

    mj_ppo.set_max_steps(1000)
    iterations = 100
    sub_batch_size = 256
    frames_per_batch = sub_batch_size * 8
    total_frames = frames_per_batch * iterations  # total frames for training
    mj_ppo.set_frames_iterations(
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        iterations=iterations,
        sub_batch_size=sub_batch_size
    )
    mj_ppo.set_lr(min_learning_rate=5e-5, max_learning_rate=1e-3)

    env = mj_ppo.setup_env_normalized(base_env)

    # create policy actor network and value critic network
    actor_network = mj_ppo.create_actor_network(env)
    probabilistic_actor_policy = mj_ppo.setup_policy(env, actor_network)
    value_module = mj_ppo.setup_value_module(env)

    # setup of replay buffer, collector
    collector = mj_ppo.create_collector(env, probabilistic_actor_policy)
    replay_buffer = mj_ppo.create_replay_buffer()

    # create data collector
    collector = mj_ppo.create_collector(env, probabilistic_actor_policy)

    # create replay buffer
    replay_buffer = mj_ppo.create_replay_buffer()

    # setup learning modules, then run the training loop
    mj_ppo.training_loop(env, probabilistic_actor_policy, value_module, replay_buffer,
                         collector, model_filepath="models/inverted_double_pendulum_ppo.pth")

    # load best policy from saved model weights
    trained_actor_policy = mj_ppo.load_policy(env, model_filepath="models/inverted_double_pendulum_ppo.pth")

    # save model program
    mj_ppo.export_policy(env, trained_actor_policy, model_filepath="models/inverted_double_pendulum_ppo.pt2")

    # initialize the base environment with rendering enabled
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode="human")
    print("Initialized rendered environment:", env_name)

    # create environment with rendering enabled for inference
    render_env = mj_ppo.setup_env(base_env)

    # run inference with the trained policy for a number of episodes
    mj_ppo.run_inference(render_env, trained_actor_policy, episodes=10)

    env.close()
    render_env.close()

def inference():
    mj_ppo.set_device()
    
    # initialize the base environment with rendering enabled
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode="human")
    print("Initialized rendered environment:", env_name)
    
    saved_model_path = "models/inverted_double_pendulum_ppo.pt2"
    
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
