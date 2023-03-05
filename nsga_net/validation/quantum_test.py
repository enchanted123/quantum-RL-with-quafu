'''
Author: jinyuxin
Date: 2022-10-25 22:28:49
Review: 2022-03-03 11:40:20
Description: Infer quantum models using stored weights.
'''

import tensorflow as tf
from sympy import im

# device
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.set_visible_devices(gpus[1], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

import sys

# update your projecty root path before running
sys.path.insert(0, ' ')
# for example, '/home/user/Documents/quantum_rl/nsga_net'

import argparse
import logging
import os
import time
from functools import reduce

import cirq
# model imports
import models.quantum_genotypes as genotypes
import numpy as np
from misc import utils
from models.quantum_models import generate_model_policy as Network
from search.quantum_train_search import gather_episodes

parser = argparse.ArgumentParser('Quantum RL Testing')
parser.add_argument('--save', type=str, default='qEXP', help='experiment name')
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument('--infer_episodes', type=int, default=10, help='the number of infer episodes')
parser.add_argument('--gamma', type=float, default=1.0, help='discount parameter')
parser.add_argument('--env_name', type=str, default="CartPole-v1", help='environment name')
parser.add_argument('--state_bounds', type=np.array, default=np.array([2.4, 2.5, 0.21, 2.5]), help='state bounds')
parser.add_argument('--n_qubits', type=int, default=4, help='the number of qubits')
parser.add_argument('--n_actions', type=int, default=2, help='the number of actions')
parser.add_argument('--arch', type=str, default='NSGANet_id10', help='which architecture to use')
parser.add_argument('--model_path', type=str, default='./weights/weights_id10_quafu.h5', help='path of pretrained model')

args = parser.parse_args(args=[])
args.save = 'infer-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

qubits = cirq.GridQubit.rect(1, args.n_qubits)
genotype = eval("genotypes.%s" % args.arch)
ops = [cirq.Z(q) for q in qubits]
observables = [reduce((lambda x, y: x * y), ops)] # Z_0*Z_1*Z_2*Z_3


def main():
    logging.info("args = %s", args)

    model = Network(qubits, genotype, args.n_actions, observables)

    model.load_weights(args.model_path)
    
    # inference 
    valid_reward = infer(model)


def infer(model):
    episode_reward_history = []
    for batch in range(args.infer_episodes // args.batch_size):
        # Gather episodes
        tasklist, episodes = gather_episodes(args.state_bounds, args.n_actions, model, args.batch_size, args.env_name, qubits, genotype)
        logging.info(tasklist)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))

        # avg_rewards = np.mean(episode_reward_history[-10:])

        logging.info('valid finished episode: %f', (batch + 1) * args.batch_size)
        logging.info('valid average rewards: %f', episode_reward_history[-1])
    
        if episode_reward_history[-1] >= 200.0:
            break
    return episode_reward_history


if __name__ == '__main__':
    main()