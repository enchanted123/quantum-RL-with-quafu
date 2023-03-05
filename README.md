# Quantum reinforcement learning with cloud Quafu
This project implements reinforcement learning(RL) based on parameterized quantum circuits(PQCs) with quantum computing cloud Quafu. The work first searches a suitable architecture through four modules including variational PQC, data-encoding PQC, entanglement and measurement as shown in the follwing figure.

![image1](https://github.com/enchanted123/quantum-RL-with-quafu/blob/main/img/4modules.png)

In this 4-qubit quantum circuit, $\mathbf{x}_1(\theta)$ refers to a variational PQC, $\mathbf{x}_2(d, \lambda)$ indicates a data-encoding PQC, $\mathbf{x}_3$ is the entanglement part and $\mathbf{x}_0(\psi)$ displays a measurement part.



The jupyter notebook in repo shows a simple example with pre-trained models.

# References
Evolutionary Quantum Architecture Search for Parametrized Quantum Circuits, L. Ding and L. Spector, arXiv:2208.11167.

NSGA-Net: Neural Architecture Search using Multi-Objective Genetic Algorithm, Z. Lu, et al, arXiv:1810.03522.

Parametrized Quantum Policies for Reinforcement Learnining, S. Jerbi, et al, arXiv:2103.05577.

https://github.com/ianwhale/nsga-net

https://tensorflow.google.cn/quantum/tutorials/quantum_reinforcement_learning
