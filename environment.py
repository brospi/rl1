import gym
import numpy as np
from gym.spaces import Box


###### ENVIRONMENT NAMES ######
# FrozenLake-v0
# FrozenLake8x8-v0
# CartPole-v1
# MountainCar-v0

class Environment:
    def __init__(self, name):
        self.name = name
        self.env = gym.make(name)

    def get_observation_space_dim(self):
        return get_observation_space_dim(self.env)

    def get_action_space_dim(self):
        return get_action_space_dim(self.env)

    def make_input(self, state):
        return make_input(self.env, state)

    def get_success_and_learning_reward(self, state, env_reward, end, dt):
        return get_success_and_learning_reward(self.env, state, env_reward, end, dt)

    def get_random_func_play(self, state):
        return self.env.action_space.sample()

    def get_move_from_graph(self, session, graph, state):
        return session.run(graph['predict'], feed_dict={graph['inputs']: self.make_input(state)})[0]


def get_observation_space_dim(env):
    if isinstance(env.observation_space, Box):
        return env.observation_space.shape[0]
    else:
        return env.observation_space.n


def get_action_space_dim(env):
    return env.action_space.n


def get_success_and_learning_reward(env, state, env_reward, end, dt):
    env_id = env.spec.id
    if env_id == 'FrozenLake8x8-v0':
        success = end and env_reward == 1
        if success:
            learning_reward = 1
        else:
            ncol = np.math.sqrt(env.observation_space.n)
            col, row = int(state) % ncol, int(state) // ncol
            rel_dist = ((ncol - col) + (ncol - row)) / (2 * ncol)
            learning_reward = rel_dist / 10.
    elif env_id == 'CartPole-v1':
        learning_reward = dt / env.spec.max_episode_steps
        success = dt == env.spec.max_episode_steps
    elif env_id == 'MountainCar-v0':
        success = end and dt < env.spec.max_episode_steps
        if success:
            learning_reward = 1
        else:
            k = 10
            area = (10 * k * ((state[0] - env.env.min_position) / (env.env.goal_position - env.env.min_position))) // k
            learning_reward = 0.02 * area
    else:
        learning_reward = env_reward
        success = end and env_reward > 0
    return success, learning_reward


def make_input(env, state):
    if isinstance(env.observation_space, Box):
        inputs = np.zeros((1, np.shape(state)[0]))
        inputs[0] = state
        return inputs
    else:
        return np.identity(env.observation_space.n)[state: state + 1]
