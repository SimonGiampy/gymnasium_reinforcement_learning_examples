import sys

# import local script with all functions
from .mujoco_rl import mujoco_ppo as mj_ppo
from .mujoco_rl.gymnasium_env_setup import setup_inference_transforms, setup_env_inference, setup_env_training
from .mujoco_rl import gymnasium_env_setup

# import Gymnasium environment wrapper
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.libs.gym import set_gym_backend

# Inverted Double Pendulum environment:
# https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/
env_name = "InvertedDoublePendulum-v5"
model_filename = "models/inverted_double_pendulum_ppo"


def training():

    mj_ppo.set_device()

    # print("list of available Gymnasium environments:", GymEnv.available_envs)

    # initialize environment for training
    with set_gym_backend("gymnasium"):
        base_env = GymEnv(env_name, device=mj_ppo.device, render_mode=None)
    print("Initialized base environment:", env_name)

    mj_ppo.set_max_steps(gymnasium_env_setup.MAX_STEPS)
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

    # setup environment with selected transforms in unbatched mode (single environment)
    env = setup_env_training(base_env)

    # create policy actor network and value critic network
    actor_network = mj_ppo.create_actor_network(env)
    probabilistic_actor_policy = mj_ppo.setup_policy(env, actor_network)
    value_module = mj_ppo.setup_value_module(env)

    # setup of replay buffer, collector
    collector = mj_ppo.create_collector(env, probabilistic_actor_policy)
    replay_buffer = mj_ppo.create_replay_buffer()

    # setup learning modules, then run the training loop
    mj_ppo.training_loop(env, probabilistic_actor_policy, value_module, replay_buffer,
                         collector, model_filepath=model_filename + ".pth")

    # initialize the base environment with rendering enabled
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode="human")
    print("Initialized rendered environment:", env_name)

    norm_float32_module = setup_inference_transforms(env=env)

    # create environment with rendering enabled for inference
    render_env = setup_env_inference(base_env)

    # load best policy from saved model weights
    trained_actor_policy = mj_ppo.load_policy_transform(env=render_env, transform_module=norm_float32_module,
                                                        model_filepath=model_filename + ".pth")

    # save model program
    mj_ppo.export_policy(env=render_env, actor_policy=trained_actor_policy,
                         model_filepath=model_filename + ".pt2")

    # run inference with the trained policy for a number of episodes
    mj_ppo.run_inference(render_env, trained_actor_policy, episodes=10)

    render_env.close()


def inference():
    mj_ppo.set_device()

    # initialize the base environment with rendering enabled
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode="human")
    print("Initialized rendered environment:", env_name)

    saved_model_path = model_filename + ".pt2"

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
