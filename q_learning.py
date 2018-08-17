from functools import partial
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from environment import Environment


class Step:
    def __init__(self, base_state, action, next_state, env_reward, end, dt):
        self.base_state = base_state
        self.action = action
        self.next_state = next_state
        self.env_reward = env_reward
        self.end = end
        self.dt = dt


def learn(session, graph, environment: Environment, params: Dict[str, Any]):
    env = environment.env
    rewards = []
    successes = []
    list_proba_random_action = []
    n_episodes = params["num_episodes"]
    for i in range(n_episodes):
        if i % 100 == 0:
            print('Ran {} episodes'.format(i))
        state = env.reset()
        dt = 0
        all_steps = []
        end = False
        while not end:
            dt += 1
            input_state = environment.make_input(state)
            action, all_q = session.run([graph['predict'], graph['Q']], feed_dict={graph['inputs']: input_state})
            action = action[0]
            if np.random.rand(1) < params['proba_random_move']:
                action = env.action_space.sample()

            next_state, env_reward, end, _ = env.step(action)
            step = Step(state, action, next_state, env_reward, end, dt)
            all_steps.append(step)

            state = next_state
            if end:
                break
        graph, success, reward = update_graph(session, environment, graph, all_steps, params)
        successes.append(success)
        rewards.append(reward)
        list_proba_random_action.append(params['proba_random_move'])
    plot_successes([np.array(successes), np.array(rewards)], additional_plots=[np.array(list_proba_random_action)])
    print("Fraction of successful episodes: {} / {}".format(sum(successes), n_episodes))
    return graph


def update_graph(session, environment: Environment, graph, all_steps: List[Step], params):
    rewards = []
    success = False
    for step in all_steps:
        success_step, learning_reward = environment.get_success_and_learning_reward(step.next_state,
                                                                               step.env_reward,
                                                                               step.end,
                                                                               step.dt)
        success = success_step or success
        rewards.append(learning_reward)
        input_from_base_state = environment.make_input(step.base_state)
        q_from_state = session.run(graph['Q'], feed_dict={graph['inputs']: input_from_base_state})
        q_from_next_state = session.run(graph['Q'], feed_dict={graph['inputs']: environment.make_input(step.next_state)})

        max_q_next_move = np.max(q_from_next_state)
        target_q = q_from_state

        target_q[0, step.action] = learning_reward + params["gamma"] * max_q_next_move

        session.run(graph['update'],
                    feed_dict={
                        graph['inputs']: input_from_base_state,
                        graph['nextQ']: target_q
                    })
    return graph, success, max(rewards)


def plot_successes(list_to_average, n=50, additional_plots=None):
    for elem in list_to_average:
        moving_average_sucesses = get_moving_average(elem, n)
        plt.plot(moving_average_sucesses)
    if additional_plots is not None:
        for additional_plot in additional_plots:
            plt.plot(additional_plot)
    plt.show()


def get_moving_average(x, n):
    moving_average = np.cumsum(x, dtype=float)
    moving_average[n:] = moving_average[n:] - moving_average[:-n]
    moving_average[n - 1:] = moving_average[n - 1:] / n
    moving_average[:n - 1] = 0
    return moving_average


def play_game_from_func(env, func):
    print("-------- Start game --------")
    state = env.reset()
    env.render()
    while True:
        action = func(state)
        state, reward, end, _ = env.step(action)
        env.render()
        if end:
            print("-------- End game --------   : {}".format('WIN' if reward == 1 else 'LOSE'))
            return reward


def get_graph(environment: Environment, sizes_hidden_layers=None, adding=False):
    tf.reset_default_graph()

    action_space_dim = environment.get_action_space_dim()
    observation_space_dim = environment.get_observation_space_dim()

    inputs1 = tf.placeholder(shape=[1, observation_space_dim], dtype=tf.float32)

    layers = [inputs1]
    if sizes_hidden_layers is not None:
        for size in sizes_hidden_layers:
            previous_layer = layers[-1]
            previous_layer_size = previous_layer.shape.dims[-1].value
            hidden_weights = tf.Variable(tf.random_uniform([previous_layer_size, size]), dtype=tf.float32)
            mul = tf.matmul(previous_layer, hidden_weights)
            if adding:
                hidden_b = tf.Variable(tf.random_uniform([size]), dtype=tf.float32)
                mul = tf.add(mul, hidden_b)
            layer = tf.nn.relu(mul)
            layers.append(layer)

    last_layer = layers[-1]
    last_layer_size = last_layer.shape.dims[-1].value
    last_weight = tf.Variable(tf.random_uniform([last_layer_size, action_space_dim], 0, 0.001))
    last_b = tf.Variable(tf.random_uniform([action_space_dim]), dtype=tf.float32)

    q_out = tf.matmul(last_layer, last_weight)
    if adding:
        q_out = tf.add(q_out, last_b)
    predict = tf.argmax(q_out, 1)

    next_q = tf.placeholder(shape=[1, action_space_dim], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(next_q - q_out))
    trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    update_model = trainer.minimize(loss)

    saver = tf.train.Saver()

    graph = {
        'Q': q_out,
        'inputs': inputs1,
        'update': update_model,
        'nextQ': next_q,
        'predict': predict,
        'saver': saver,
    }

    return graph


def get_policy(session, environment, graph):
    policy = {}
    for state in range(environment.get_observation_space_dim()):
        action = environment.get_move_from_graph(session, graph, state)
        policy[state] = action
    return policy


def execute(name, restore=False):
    environment = Environment(name)
    params = {
        'gamma': 0.99,
        'num_episodes': 1000,
        'proba_random_move': 0.1
    }
    graph = get_graph(environment, sizes_hidden_layers=[])
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    filename = './tf_save_q_learning/{}.ckpt'.format(environment.name)
    if restore:
        graph['saver'].restore(session, filename)
    # play_game_from_func(environment.env, environment.get_random_func_play)
    # play_game_from_func(environment.env, partial(environment.get_move_from_graph, session, graph))
    graph = learn(session, graph, environment, params)
    # play_game_from_func(environment.env, partial(environment.get_move_from_graph, session, graph))
    saver_path = graph['saver'].save(session, filename)
    print("Saved graph in {}".format(saver_path))


if __name__ == '__main__':
    while True:
        execute("MountainCar-v0", True)
