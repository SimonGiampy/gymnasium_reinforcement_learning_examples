# import local script with all functions
import mujoco_ppo as mj_ppo

# import Gymnasium environment wrapper
from torchrl.envs.libs.gym import GymEnv

# Inverted Double Pendulum environment:
# https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/
env_name = "InvertedDoublePendulum-v5"

def main():
    
    print("list of available Gymnasium environments:", GymEnv.available_envs)

    # initialize environment for training
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode=None)
    print("Initialized base environment:", env_name)
    
    mj_ppo.set_max_steps(1000)
    mj_ppo.set_iterations(100)
    
    env = mj_ppo.setup_env_normalized(base_env)

    # create policy actor network and value critic network
    actor_network = mj_ppo.create_actor_network(env)
    probabilistic_actor_policy = mj_ppo.setup_policy(env, actor_network)
    value_module = mj_ppo.setup_value_module(env)

    # print("Running policy actor module:", probabilistic_actor_policy(env.reset()))
    # print("Running value critic module:", value_module(env.reset()))

    # setup of replay buffer, collector
    collector = mj_ppo.create_collector(env, probabilistic_actor_policy)
    replay_buffer = mj_ppo.create_replay_buffer()

    # create data collector
    collector = mj_ppo.create_collector(env, probabilistic_actor_policy)

    # create replay buffer
    replay_buffer = mj_ppo.create_replay_buffer()

    # setup learning modules, then run the training loop
    mj_ppo.training_loop(env, probabilistic_actor_policy, value_module, replay_buffer, collector)

    # load best policy from saved model weights
    trained_actor_policy = mj_ppo.load_best_policy(env)

    # save model program
    mj_ppo.export_policy(env, trained_actor_policy)
    
    # initialize the base environment with rendering enabled
    base_env = GymEnv(env_name, device=mj_ppo.device, render_mode="human")
    print("Initialized rendered environment:", env_name)
    
    # create environment with rendering enabled for inference
    render_env = mj_ppo.setup_env(base_env)

    # run inference with the trained policy for a number of episodes
    mj_ppo.run_inference(render_env, trained_actor_policy, episodes=10)


if __name__ == "__main__":
    main()
