# Quantum reinforcement learning with cloud Quafu
This project implements reinforcement learning(RL) based on parameterized quantum circuits(PQCs) with quantum computing cloud Quafu. The work first searches a suitable architecture through four modules including variational PQC, data-encoding PQC, entanglement and measurement as shown in the follwing figure.

![image1](https://github.com/enchanted123/quantum-RL-with-quafu/blob/main/img/4modules.png)

In this 4-qubit quantum circuit, $\mathbf{x}_1(\theta)$ refers to a variational PQC, $\mathbf{x}_2(d, \lambda)$ indicates a data-encoding PQC, $\mathbf{x}_3$ is the entanglement part and $\mathbf{x}_0(\psi)$ displays a measurement part. 

During evolutionary process, multi-objective is set to gain higher performance in RL task and reduce entanglement to achieve lower depth of compiled circuits.

Then, according to the effective architecture selected, one can train the model with Cirq simulator or Quafu backend. This file provides pre-trained model in './weights' of both ways and visualize them with animation in './visualization'. Noticeablely, we support three different environments including CartPole, MountainCar and Acrobot.

| CartPole | MountainCar | Acrobot |
|:-------------:|:-------------:|:-------------:|
| ![Image2](https://github.com/enchanted123/quantum-RL-with-quafu/blob/main/img/gym_CartPole_96.gif) | ![Image3](https://github.com/enchanted123/quantum-RL-with-quafu/blob/main/img/gym_MC_PPO.gif) | ![Image4](https://github.com/enchanted123/quantum-RL-with-quafu/blob/main/img/gym_AB_PPO.gif) |

# Usage
If you want to use the pre-trained model directly, the jupyter notebook in repo shows how to interact with Quafu and can output a gif if you have an access to display or you can refer to test code in './validation' to infer pre-trained model and get corresponding records in a 'log.txt' or '.csv'.

If you want to train the model from scratch, you may need download this repo and run the whole process, it's important to get your own token by registering on http://quafu.baqis.ac.cn/ .

# Updates

Now, we support more reinforcement learning algorithms (policy gradient and hybrid quantum-classical ppo) and environments (CartPole, MountainCar and Acrobot). All the algorithms and environments can be tested on simulator and quafu cloud.

# References
Evolutionary Quantum Architecture Search for Parametrized Quantum Circuits, L. Ding and L. Spector, arXiv:2208.11167.

NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm, Z. Lu, et al, arXiv:1810.03522.

Parametrized Quantum Policies for Reinforcement Learnining, S. Jerbi, et al, arXiv:2103.05577.

https://github.com/ianwhale/nsga-net

https://tensorflow.google.cn/quantum/tutorials/quantum_reinforcement_learning

https://github.com/ScQ-Cloud/pyquafu

# Article

**Quafu-RL: The Cloud Quantum Computers based Quantum Reinforcement Learning,** BAQIS Quafu Group, arXiv:2305.17966.
