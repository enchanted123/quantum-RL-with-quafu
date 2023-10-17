# Note: If you want to train the model from scratch, please use the command 'pip install gym==0.26.0' first.
import argparse
import logging
import os
import sys


import time
from functools import reduce
from tensorflow.keras.optimizers import Adam
import cirq
import gym
import models.quantum_genotypes as genotypes
import numpy as np
import tensorflow as tf
from misc.utils import compute_returns, create_exp_dir, gather_episodes, gather_episodes_, train
from models.quantum_models import generate_model_policy as Network, generate_model_policy
from models.value_net import ValueNet



parser = argparse.ArgumentParser('Quantum RL Training')
parser.add_argument('--save', type=str, default='PPO_TRAIN', help='experiment name')
parser.add_argument('--epochs', type=int, default=10, help='batch size in one train')
parser.add_argument('--n_episodes', type=int, default=1000, help='the number of episodes')
# parser.add_argument('--infer_episodes', type=int, default=5, help='the number of infer episodes')
parser.add_argument('--gamma', type=float, default=0.98, help='gamma for GME')
parser.add_argument('--lmbda', type=float, default=0.95, help='lmbda for GME')
parser.add_argument('--eps', type=float, default=0.2, help='eps for PPO')
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='environment name')
parser.add_argument('--state_bounds', type=np.array, default=np.array([2.4, 2.5, 0.21, 2.5]), help='state bounds')
parser.add_argument('--n_qubits', type=int, default=4, help='the number of qubits')
parser.add_argument('--n_actions', type=int, default=2, help='the number of actions')
parser.add_argument('--arch', type=str, default='ORI_TYPE_CP', help='which architecture to use')
parser.add_argument('--lr_in', type=float, default=0.1, help='learning rate of input parameter')
parser.add_argument('--lr_var', type=float, default=0.01, help='learning rate of variational parameter')
parser.add_argument('--lr_out', type=float, default=0.1, help='learning rate of output parameter')
parser.add_argument('--lr_critic', type=float, default=0.1, help='learning rate of critic')
parser.add_argument('--beta', type=float, default=1.0, help='output parameter')
# parser.add_argument('--model_path', type=str, default='./weights/train_p18/weights_id10_quafu_86.h5',
#                     help='path of pretrained model')
# parser.add_argument('--backend', type=str, default='quafu', help='choose cirq simulator or quafu cloud platform')
# parser.add_argument('--shots', type=int, default=1000, help='the number of sampling')
# parser.add_argument('--backend_quafu', type=str, default='ScQ-P10', help='which quafu backend to use')

args = parser.parse_args(args=[])
args.save = 'train-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main(qubits, genotype, observables):
    logging.info("args = %s", args)

    # model = Network(qubits, genotype, args.n_actions, args.beta, observables, args.env_name)

    actor = generate_model_policy(qubits, genotype, args.n_actions, args.beta, observables, args.env_name, using_H=True)

    critic = ValueNet()  # 价值网络
    # model.load_weights(args.model_path)
    state_ub = args.state_bounds
    state_lb = -state_ub

    optimizer_in = Adam(learning_rate=args.lr_in, amsgrad=False, epsilon=1e-5)
    optimizer_var = Adam(learning_rate=args.lr_var, amsgrad=False, epsilon=1e-5)
    optimizer_out = Adam(learning_rate=args.lr_out, amsgrad=False, epsilon=1e-5)



    # critic_optimizer = Adam(learning_rate=critic_lr)
    critic_optimizer = Adam(learning_rate=args.lr_critic, amsgrad=False, epsilon=1e-5)  # 这个不知道 似乎作用有限

    return_list = train(args.env_name, actor, critic, args.gamma, args.lmbda, args.eps, args.epochs,
                        optimizer_in, optimizer_var, optimizer_out, critic_optimizer, 500.,
                        critic_normalized=True, state_ub=state_ub, state_lb=state_lb, num_episodes=args.n_episodes)





if __name__ == '__main__':
    qubits = cirq.GridQubit.rect(1, args.n_qubits)
    genotype = eval("genotypes.%s" % args.arch)
    ops = [cirq.Z(q) for q in qubits]
    observables = [reduce((lambda x, y: x * y), ops)]  # Z_0*Z_1*Z_2*Z_3

    main(qubits, genotype, observables)
