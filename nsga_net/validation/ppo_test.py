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
from misc.utils import compute_returns, create_exp_dir, gather_episodes, gather_episodes_, train, normalized_state
from models.quantum_models import generate_model_policy as Network, generate_model_policy
from models.value_net import ValueNet



parser = argparse.ArgumentParser('Quantum RL Training')
parser.add_argument('--save', type=str, default='PPO_TEST', help='experiment name')
parser.add_argument('--epochs', type=int, default=10, help='batch size in one train')
parser.add_argument('--infer_episodes', type=int, default=100, help='the number of infer episodes')
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
parser.add_argument('--model_path', type=str, default='../weights/example/ppo/ppo_simulator/all_state_normalize/weights_CP_ACTOR_NO_amsgrad.h5', help='path of pretrained model')
# parser.add_argument('--model_path', type=str, default='./weights/train_p18/weights_id10_quafu_86.h5',
#                     help='path of pretrained model')
# parser.add_argument('--backend', type=str, default='quafu', help='choose cirq simulator or quafu cloud platform')
# parser.add_argument('--shots', type=int, default=1000, help='the number of sampling')
# parser.add_argument('--backend_quafu', type=str, default='ScQ-P10', help='which quafu backend to use')

args = parser.parse_args(args=[])
args.save = 'infer-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
create_exp_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

qubits = cirq.GridQubit.rect(1, args.n_qubits)
genotype = eval("genotypes.%s" % args.arch)
ops = [cirq.Z(q) for q in qubits]
observables = [reduce((lambda x, y: x * y), ops)]  # Z_0*Z_1*Z_2*Z_3


def main():
    logging.info("args = %s", args)

    model = generate_model_policy(qubits, genotype, args.n_actions, args.beta, observables, args.env_name, using_H=True)

    model.load_weights(args.model_path)

    # inference
    valid_reward = infer(model)


def infer(model):
    episode_reward_history = []
    for batch in range(args.infer_episodes):
        # Gather episodes
        state_ub = args.state_bounds
        state_lb = -state_ub

        env = gym.make(args.env_name)

        state, _ = env.reset()

        norm_state = normalized_state(state, state_ub, state_lb)  #

        done = False
        episode_return = 0
        while not done:
            # actor根据策略采取行动
            probs = model(tf.convert_to_tensor([norm_state]))
            probs = probs[0].numpy()
            action = np.random.choice(range(len(probs)), p=probs)
            # 记录
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # gym=0.26.x
            episode_return += reward
            norm_state = normalized_state(next_state, state_ub, state_lb)  #
        env.close()
        episode_reward_history.append(episode_return)
        # Store collected rewards

        # avg_rewards = np.mean(episode_reward_history[-10:])

        logging.info('valid average rewards: %f', episode_reward_history[-1])

        # if episode_reward_history[-1] >= 200.0:
        #     break
    return episode_reward_history


if __name__ == '__main__':
    main()