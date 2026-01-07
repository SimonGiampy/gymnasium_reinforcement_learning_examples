# Deep Reinforcement Learning samples using TorchRL

This repository contains sample code for Deep Reinforcement Learning (DRL) using the **TorchRL library**.
The samples demonstrate how to set up and train DRL agents in various environments, including MuJoCo simulations.

The code provides implementation of DRL algorithms applied to different tasks, showcasing the capabilities of TorchRL for building and training reinforcement learning agents, with interfaces and wrappers that simplify the interaction with popular environments.

This project is intended for educational purposes and as a starting point for DRL with TorchRL.

Author:
- [Simone Giampà](https://github.com/simongiampy)

## Installation

Create a virtual python environment and install the required packages:

```bash
pip3 install pytorch torchrl gymnasium[classic_control] gymnasium[mujoco] dm_control
```

Before running the Mujoco samples, ensure you have Mujoco installed and configured properly.

These dependencies also require a working installation of CUDA if you plan to run the samples on a GPU.

## Algorithms

### Proximal Policy Optimization (PPO)

The PPO algorithm implementation in [ppo.py](ppo/ppo.py) demonstrates how to implement the algorithm from scratch using PyTorch.
Its implementation is derived from [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/algorithms/ppo.html) by OpenAI.
The source code is adapted from the [original repository](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo), with added comments and formulas for explanation and improved readability.

Useful links and reads:
- [PPO from OpenAI Spinning UP](https://spinningup.openai.com/en/latest/algorithms/ppo.html)
- [PPO integration with TorchRL tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
- [Proximal Policy Optimization Algorithms paper](https://arxiv.org/abs/1707.06347)

### Deep Q-Network (DQN)

The DQN and Double DQN algorithms implementation is found integrated in the [cartpole_dqn.py](dqn/cartpole_dqn.py) sample.
The implementation is inspired by the original DQN paper by DeepMind and adapted to work with PyTorch.
This script is derived directly from the [TorchRL DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

Useful links and reads:
- [DQN Tutorial from PyTorch](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Human-level control through deep reinforcement learning paper](https://www.nature.com/articles/nature14236)


## Mujoco Simulations

Mujoco is a physics engine widely used for simulating robotic systems and control tasks.
The samples in this section demonstrate how to set up and train DRL agents in Mujoco environments using the PPO algorithm.

Check out the complete tutorial, adapted from the [TorchRL Mujoco PPO tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html), in the Jupyter Notebook [tutorial_mujoco_ppo.ipynb](tutorial_mujoco_ppo.ipynb) file.

The main code for the integration of PPO algorithm, with Mujoco environments, wrapped in TorchRL environments, with a format suitable for training, similar to the Gymnasium wrapper, is found in the [mujoco_rl/mujoco_ppo.py](mujoco_rl/mujoco_ppo.py) file.


### Gymnasium Environments

Gymnasium is a popular library for developing and comparing reinforcement learning algorithms.
It is conveniently integrated with TorchRL through environment wrappers. 
The wrapper allows also to easily parallelize multiple environment instances for faster data collection.

#### Inverted Double Pendulum

This environment involves balancing a double pendulum in an upright position by applying forces to the base.
It is described in detail in the [Gymnasium Inverted Double Pendulum documentation](https://gymnasium.farama.org/environments/mujoco/inverted_double_pendulum/).

It is implemented with the python script [inverted_double_pendulum_mujoco.py](inverted_double_pendulum_mujoco.py).
It uses the PPO algorithm implementation from [mujoco_ppo.py](mujoco_rl/mujoco_ppo.py), since
it is a continuous action space environment.

#### Walking Humanoid

This environment involves controlling a humanoid robot to walk forward as fast as possible without falling.
It is described in detail in the [Gymnasium Humanoid documentation](https://gymnasium.farama.org/environments/mujoco/humanoid/).

It is implemented with the python script [humanoid_mujoco.py](humanoid_gymnasium_mujoco.py).
It also uses the PPO algorithm implementation from [mujoco_ppo.py](mujoco_rl/mujoco_ppo.py).

### DeepMind Control Suite Environments

DeepMind Control Suite is a collection of continuous control tasks for benchmarking reinforcement learning algorithms.
It is integrated with TorchRL through environment wrappers, similar to Gymnasium.
This suite of environments provides a set of challenging tasks for evaluating the performance of RL agents,
all implemented with the MuJoCo physics engine.

The script [deepmind_control_mujoco.py](deepmind_control_mujoco.py) demonstrates how to set up and train a PPO agent in a DeepMind Control Suite environment using TorchRL. It can be used across all environments and tasks available in the suite.

## Useful Resources

### TorchRL Library

TorchRL is a powerful library for reinforcement learning in PyTorch. 
It provides a wide range of tools and functionalities to facilitate the development and training of RL agents.
Some of the functionalities used in this repository include:
- Environment wrappers for Gymnasium and DeepMind Control Suite
- Data collectors for gathering experience from environments
- Replay buffers for storing and sampling experience
- Loss functions for multiple RL algorithms

Useful links:
- [TorchRL Documentation](https://docs.pytorch.org/rl/stable/index.html)
- [TorchRL Introduction Tutorial](https://docs.pytorch.org/rl/main/tutorials/torchrl_demo.html)
- [TorchRL Mujoco DQN tutorial](https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [TorchRL Mujoco PPO tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
- [TorchRL SuperMario DDQN tutorial](https://docs.pytorch.org/tutorials/intermediate/mario_rl_tutorial.html)

### Gymnasium Library

Gymnasium is a popular library for developing and comparing reinforcement learning algorithms.
It provides a wide variety of environments, including classic control tasks, Atari games, and robotic simulations.
The documentation includes guides on how to use Gymnasium environments and integrate them with RL algorithms.
TorchRL provides wrappers to easily interface Gymnasium environments with RL agents.

Useful links:
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Gymnasium GitHub Repository](https://github.com/Farama-Foundation/Gymnasium)
- [Gymnasium Intro Tutorial](https://gymnasium.farama.org/introduction/basic_usage/)

### MuJoCo Physics Engine

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine designed for simulating complex robotic systems and control tasks.
It is widely used in the reinforcement learning community for benchmarking algorithms on continuous control tasks.

Mujoco XLA is a version of MuJoCo that leverages XLA (Accelerated Linear Algebra) for improved performance on modern hardware, including GPUs and TPUs. It supports JAX for automatic differentiation and efficient computation.

Mujoco WARP is another variant of MuJoCo that focuses on high-performance simulation using the NVIDIA WARP framework. 
It is designed to provide extremely parallelizable and differentiable simulations for robotics and reinforcement learning applications.

