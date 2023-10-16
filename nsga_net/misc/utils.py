import os
import re
import shutil
from collections import defaultdict

import gym
import numpy as np
import tensorflow as tf
from models.quantum_models import generate_circuit, get_model_circuit_params
from quafu import QuantumCircuit as quafuQC, QuantumCircuit, simulate
from quafu import Task, User
from gym.spaces import Discrete


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def get_res_exp(res):
    """
    Access to probabilities of all possibilities, observable specifies to Z_0*Z_1*Z_2*Z_3.
    """
    prob = res.probabilities
    sumexp = 0
    for k, v in prob.items():
        count = 0
        for i in range(len(k)):
            if k[i] == '1':
                count += 1
        if count % 2 == 0:
            sumexp += v
        else:
            sumexp -= v
    return sumexp


def get_quafu_exp(circuit, n_qubits, backend_quafu, shots):
    """
    Execute circuits on quafu cloud platform and return the expectation.
    """
    # convert Cirq circuts to qasm
    openqasm = circuit.to_qasm(header='')
    openqasm = re.sub('//.*\n', '', openqasm)
    openqasm = "".join([s for s in openqasm.splitlines(True) if s.strip()])
    
    # fill in with your token, register on website http://quafu.baqis.ac.cn/
    user = User()
    user.save_apitoken(" ")
    
    # initialize to Quafu circuits
    q = quafuQC(n_qubits)
    q.from_openqasm(openqasm)
    
    # create the task
    task = Task()
      
    task.config(backend_quafu, shots, compile=True, priority=3)
    task_id = task.send(q, wait=True).taskid
    print('task_id:', task_id)
    
    # retrieve the result of completed tasks and compute expectations
    task_status = task.retrieve(task_id).task_status
    if task_status == 'Completed':
        task = Task()
        res = task.retrieve(task_id)
        OB = get_res_exp(res)
    return task_id, tf.convert_to_tensor([[OB]])


def get_quafu_exp_(circuit, n_qubits, backend_quafu, shots):
    """
        Execute circuits on quafu cloud platform and return the expectation.
    """
    # 2 转化为 openqasm
    openqasm = circuit.to_qasm(header='')
    openqasm = re.sub('//.*\n', '', openqasm)
    openqasm = "".join([s for s in openqasm.splitlines(True) if s.strip()])

    # 3 利用openqasm转化为 quafu 可以测量的电路
    quafu_qc = QuantumCircuit(n_qubits)
    quafu_qc.from_openqasm(openqasm)

    # # 假假
    #
    # # layer_ = actor.get_layer("re-uploading_PQC")
    # # out.append(layer_(inputs))
    # # # # 假4
    # # 通过夸父模拟器看看 这个 夸父算期望的时候 有什么不一样的地方
    simu_res = simulate(quafu_qc)
    # ob = simu_res.calculate_obs([0, 1, 2, 3])
    # taskid = 'test'
    # # return 'test', tf.convert_to_tensor([[ob]], dtype=tf.float32)

    # # 真4
    task = Task()
    task.config(backend=backend_quafu, shots=shots, compile=True)
    taskid = task.send(quafu_qc, wait=True).taskid

    res = task.retrieve(taskid)
    ob = res.calculate_obs([0, 1, 2, 3])
    if res.task_status == 'Completed' and not np.isnan(ob).any():
        print(f'quafu正在执行{taskid}, res={ob} 夸父模拟器值={simu_res.calculate_obs([0, 1, 2, 3])}', end='  ')
        # print(f'夸父模拟器值={simu_res.calculate_obs([0, 1, 2, 3])}', end='')

        return taskid, ob
    else:
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print(f'quafu正在执行{taskid}, res={ob} 夸父模拟器值={simu_res.calculate_obs([0, 1, 2, 3])}', end='  ')
        ob = simu_res.calculate_obs([0, 1, 2, 3])
    return taskid, ob


def get_compiled_gates_depth(circuit, n_qubits, backend_quafu, shots):
    """
    Get the gates and layered circuits of compiled circuits.
    """
    openqasm = circuit.to_qasm(header='')
    openqasm = re.sub('//.*\n', '', openqasm)
    openqasm = "".join([s for s in openqasm.splitlines(True) if s.strip()])
    
    user = User()
    user.save_apitoken(" ")
    
    q = quafuQC(n_qubits)
    q.from_openqasm(openqasm)
    
    task = Task()
    
    task.config(backend_quafu, shots, compile=True)
    task_id = task.send(q, wait=True).taskid
    print('task_id:', task_id)
    
    task_status = task.retrieve(task_id).task_status
    if task_status == 'Completed':
        task = Task()
        res = task.retrieve(task_id)
        gates = res.transpiled_circuit.gates
        layered_circuit = res.transpiled_circuit.layered_circuit()
    return task_id, gates, layered_circuit


class Alternating_(tf.keras.layers.Layer):
    """
    Load observable weights of pre-trained models.
    """
    def __init__(self, obsw):
        super(Alternating_, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant(obsw), dtype="float32", trainable=True, name="obsw")

    def call(self, inputs):
        return tf.matmul(inputs, self.w)


def get_obs_policy(obsw, beta):
    """
    Output the final policy.
    """
    process = tf.keras.Sequential([ Alternating_(obsw),
                                    tf.keras.layers.Lambda(lambda x: x * beta),
                                    tf.keras.layers.Softmax()
                                ], name="obs_policy")
    return process


def get_height(position):
    """
    Get the height of position in MountainCar-v0.
    """
    return np.sin(3 * position)*.45+.55


def gather_episodes(state_bounds, n_actions, model, n_episodes, env_name, beta, backend, backend_quafu='ScQ-P10', shots=1000, 
                    n_qubits=4, qubits=None, genotype=None):
    """
    Interact with environment, you can choose the backend between `cirq` simulator and `quafu` cloud platform.
    """
    # trajectories = [defaultdict(list) for _ in range(n_episodes)]
    # envs = [gym.make(env_name) for _ in range(n_episodes)]
    #
    # done = [False for _ in range(n_episodes)]
    # states = [e.reset() for e in envs]

    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]

    done = [False for _ in range(n_episodes)]
    # states = [e.reset() for e in envs]
    states = [e.reset()[0] for e in envs]

    tasklist = []

    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        normalized_states = [s/state_bounds for i, s in enumerate(states) if not done[i]]
        # height = [get_height(s[0]) for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]['states'].append(state)

        # Compute policy for all unfinished envs in parallel
        states = tf.convert_to_tensor(normalized_states)

        if backend == 'cirq':
            action_probs = model([states])
        elif backend == 'quafu':
            newtheta, newlamda = get_model_circuit_params(qubits, genotype, model)
            circuit, _, _ = generate_circuit(qubits, genotype, newtheta, newlamda, states.numpy()[0])
            taskid, expectation = get_quafu_exp(circuit, n_qubits, backend_quafu, shots)
            tasklist.append(taskid)
            # print('gather_episodes_exp:', expectation)

            obsw = model.get_layer('observables-policy').get_weights()[0]
            obspolicy = get_obs_policy(obsw, beta)
            action_probs = obspolicy(expectation)
        else:
            print('This backend is not supported now.')

        # Store action and transition all environments to the next state
        states = [None for i in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):
            trajectories[i]['action_probs'].append(policy)
            action = np.random.choice(n_actions, p=policy)
            # states[i], reward, done[i], _ = envs[i].step(action)
            states[i], reward, terminated, truncated, _ = envs[i].step(action)
            done[i] = terminated or truncated
            trajectories[i]['actions'].append(action)
            if env_name == "CartPole-v1":
                trajectories[i]['rewards'].append(reward)
            elif env_name == "MountainCar-v0":
                trajectories[i]['rewards'].append(reward + get_height(states[i][0]))
            else:
                print('This environment is not supported now.')

    return tasklist, trajectories


def compute_returns(rewards_history, gamma):
    """
    Compute discounted returns with discount factor `gamma`.
    """
    returns = []
    discounted_sum = 0
    for r in rewards_history[::-1]:
        discounted_sum = r + gamma * discounted_sum
        returns.insert(0, discounted_sum)

    # Normalize them for faster and more stable learning
    returns = np.array(returns)
    returns = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    returns = returns.tolist()

    return returns


def gather_episodes_(state_ub, state_lb, n_actions, model, n_episodes, env_name, backend='cirq', backend_quafu='ScQ-P10',
                    shots=1000,
                    n_qubits=4, qubits=None, genotype=None):
    """Interact with environment in batched fashion."""

    trajectories = [defaultdict(list) for _ in range(n_episodes)]
    envs = [gym.make(env_name) for _ in range(n_episodes)]

    done = [False for _ in range(n_episodes)]
    # states = [e.reset() for e in envs]
    states = [e.reset()[0] for e in envs]

    tasklist = []

    while not all(done):
        unfinished_ids = [i for i in range(n_episodes) if not done[i]]
        # normalized_states = [s / state_bounds for i, s in enumerate(states) if not done[i]]
        normalized_states = [normalized_state(s, ub=state_ub, lb=state_lb) for i, s in enumerate(states) if not done[i]]
        # height = [get_height(s[0]) for i, s in enumerate(states) if not done[i]]

        for i, state in zip(unfinished_ids, normalized_states):
            trajectories[i]['states'].append(state)

        # Compute policy for all unfinished envs in parallel
        states = tf.convert_to_tensor(normalized_states)

        if backend == 'cirq':
            action_probs = model([states])
        elif backend == 'quafu':
            # new_theta, new_lamda = get_model_circuit_params(qubits, genotype, model)
            new_theta, new_lamda = model.get_layer('nsganet_PQC').get_weights()

            expectation = []
            for i in range(int(tf.gather(tf.shape(states), 0))):
                circuit, _, _ = generate_circuit(qubits, genotype, new_theta[0], new_lamda, states.numpy()[i], using_H=True)
                taskid, ob = get_quafu_exp_(circuit, n_qubits, backend_quafu, shots)

                tasklist.append(taskid)
                expectation.append([ob])
                real_layer = model.get_layer('nsganet_PQC')
                print(f'  cirq模拟器={real_layer([tf.convert_to_tensor([states.numpy()[i]])])}')
            obs_layer = model.get_layer('observables-policy')
            action_probs = obs_layer(tf.convert_to_tensor(expectation, dtype=tf.float32))
            print(f'{action_probs} vs {model([states])}')


            # print(action_probs - model([states]))

        else:
            raise ValueError("This backend is not supported now.")

        # Store action and transition all environments to the next state
        states = [None for _ in range(n_episodes)]
        for i, policy in zip(unfinished_ids, action_probs.numpy()):

            trajectories[i]['action_probs'].append(policy)  #

            action = np.random.choice(n_actions, p=policy)
            states[i], reward, terminated, truncated, _ = envs[i].step(action)
            done[i] = terminated or truncated
            trajectories[i]['actions'].append(action)

            if env_name == 'MountainCar-v0':
                trajectories[i]['rewards'].append(reward + get_height(states[i][0]))
            else:
                trajectories[i]['rewards'].append(reward)

    return tasklist, trajectories


def normalized_state(s, ub, lb):
    return -1 + 2 * (s - lb) / (ub - lb)


# 计算优势函数
def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.numpy()
    advantage_list = []
    advantage = 0.0

    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)

    advantage_list.reverse()
    advantage_list = np.array(advantage_list)
    advantage_list = (advantage_list - np.mean(advantage_list)) / (
            np.std(advantage_list) + 1e-8
    )  # 标准化
    return tf.convert_to_tensor(advantage_list, dtype=tf.float32)


def train(env_name, actor, critic, gamma, lmbda, eps, epochs,
          optimizer_in, optimizer_var, optimizer_out, critic_optimizer, stop_avg,
          state_ub, state_lb, critic_normalized=True, num_episodes=1000):
    # num_episodes = 1000
    return_list = []

    # state_bounds = np.array([2.4, 2.5, 0.21, 2.5])
    # state_ub = state_bounds
    # state_lb = -1 * state_bounds
    env = gym.make(env_name)
    # 获取动作空间的维度
    if isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n
    else:
        raise Exception("不支持该环境！")

    # Assign the model parameters to each optimizer
    w_in, w_var, w_out = 1, 0, 2

    for cur_episode in range(num_episodes):
        # sample_one_episode_history
        episode_return = 0
        history_dict = {
            "states": [],
            "norm_state": [],
            "norm_next_states": [],
            "actions": [],
            "next_states": [],
            "rewards": [],
            "dones": [],
        }
        state, _ = env.reset()  # gym=0.26.x
        norm_state = normalized_state(state, state_ub, state_lb)  #
        # if critic_normalized:
        #     state = normalized_state(state, state_ub, state_lb)  #
        #     norm_state = state  #
        # else:
        #     norm_state = normalized_state(state, state_ub, state_lb)  #

        done = False
        while not done:
            # actor根据策略采取行动
            probs = actor(tf.convert_to_tensor([norm_state]))
            probs = probs[0].numpy()
            action = np.random.choice(range(len(probs)), p=probs)
            # 记录
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # gym=0.26.x

            if env_name == "MountainCar-v0":  # 注意 is 是身份比较
                reward = reward + get_height(state[0])

            history_dict["states"].append(state)
            history_dict["norm_state"].append(norm_state)
            history_dict["actions"].append(action)
            history_dict["next_states"].append(next_state)
            history_dict["rewards"].append([reward])
            history_dict["dones"].append([done])

            state = next_state
            norm_state = normalized_state(next_state, state_ub, state_lb)  #
            history_dict["norm_next_states"].append(norm_state)

            episode_return += reward

        return_list.append(episode_return)
        avg = np.mean(return_list[-10:])
        print(f"当前迭代次数为：{cur_episode}, 当前奖励为：{episode_return}, 平均奖励为：{avg}")
        if avg >= stop_avg:
            break

        # update_in_one_episode

        # prepare data
        states = tf.convert_to_tensor(history_dict["states"], dtype=tf.float32)
        norm_states = tf.convert_to_tensor(history_dict["norm_state"], dtype=tf.float32)
        actions = tf.convert_to_tensor(history_dict["actions"], dtype=tf.int32)
        rewards = tf.convert_to_tensor(history_dict["rewards"], dtype=tf.float32)
        next_states = tf.convert_to_tensor(history_dict["next_states"], dtype=tf.float32)

        norm_next_states = tf.convert_to_tensor(history_dict["norm_next_states"], dtype=tf.float32)

        dones = tf.convert_to_tensor(history_dict["dones"], dtype=tf.float32)

        if critic_normalized:
            td_target = rewards + gamma * critic(norm_next_states) * (1 - dones)
            td_delta = td_target - critic(norm_states)
        else:
            td_target = rewards + gamma * critic(next_states) * (1 - dones)
            td_delta = td_target - critic(states)

        advantage = compute_advantage(gamma, lmbda, td_delta)  # 广义优势函数
        old_log_probs = tf.math.log(
            tf.expand_dims(
                tf.boolean_mask(
                    actor(norm_states),
                    tf.one_hot(actions, depth=action_dim, dtype=tf.int32),
                ),
                axis=1,
            )
        )

        for _ in range(epochs):
            # update policy_net
            with tf.GradientTape() as tape_policy:
                log_probs = tf.math.log(
                    tf.expand_dims(
                        tf.boolean_mask(
                            actor(norm_states),
                            tf.one_hot(actions, depth=action_dim, dtype=tf.int32),
                        ),
                        axis=1,
                    )
                )
                ratio = tf.math.exp(log_probs - old_log_probs)  # exp(ln a - ln b) = a/b
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1 - eps, 1 + eps) * advantage  # 截断
                actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))  # PPO损失函数
                policy_gradients = tape_policy.gradient(
                    actor_loss, actor.trainable_variables
                )

                # 使用梯度裁剪
                clip_norm = .5
                # normed_policy_gradients, _ = tf.clip_by_norm(policy_gradients, clip_norm)
                # Apply some clipping
                normed_policy_gradients = [tf.clip_by_norm(g, clip_norm) for g in policy_gradients]

            for optimizer, w in zip([optimizer_in, optimizer_var, optimizer_out], [w_in, w_var, w_out]):
                optimizer.apply_gradients([(normed_policy_gradients[w], actor.trainable_variables[w])])

            # update value_net
            with tf.GradientTape() as tape_value:
                critic_loss = tf.reduce_mean(
                    tf.losses.mean_squared_error(
                        critic(norm_states) if critic_normalized else critic(states),
                        td_target)
                )
                value_gradients = tape_value.gradient(
                    critic_loss, critic.trainable_variables
                )
                # 使用梯度裁剪
                clip_norm = .5
                # normed_policy_gradients, _ = tf.clip_by_norm(policy_gradients, clip_norm)
                # Apply some clipping
                normed_value_gradients = [tf.clip_by_norm(g, clip_norm) for g in value_gradients]

            critic_optimizer.apply_gradients(
                grads_and_vars=zip(normed_value_gradients, critic.trainable_variables)
            )
    return return_list