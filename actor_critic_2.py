# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69
import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque


# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.gamma = .95
        self.tau = .125

        self.memory = deque(maxlen=100)
        self.actor_model = self.create_actor_model()
        self.target_actor_model = self.create_actor_model()

        self.critic_model = self.create_critic_model()
        self.target_critic_model = self.create_critic_model()

        self.target_reward_model = self.create_reward_model(self.target_actor_model, self.target_critic_model)
        trainer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.optimize = trainer.minimize(tf.negative(self.target_reward_model.output))

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def create_reward_model(self, actor_model, critic_model):
        state_input = Input(shape=self.env.observation_space.shape)
        action = actor_model(state_input)
        reward = critic_model([state_input, action])

        reward_model = Model(input=state_input, output=reward)
        reward_model.compile(loss="mse", optimizer='rmsprop')
        return reward_model

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)

        model = Model(input=state_input, output=output)
        model.compile(loss="mse", optimizer='rmsprop')
        return model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return model

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def forget_all(self):
        self.memory.clear()

    def _train_actor(self, samples):
        for sample in samples:
            cur_state = sample[0]
            self.sess.run(self.optimize,
                          feed_dict={
                              self.target_reward_model.input: cur_state,
                          })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict([new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def train_old(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    def train(self, clear_after=False):
        samples = self.memory
        self._train_critic(samples)
        self._train_actor(samples)
        if clear_after:
            self.forget_all()

    def _update_actor_target(self):
        self._update(self.actor_model, self.target_actor_model)

    def _update_critic_target(self):
        self._update(self.critic_model, self.target_critic_model)

    def _update(self, base, target):
        base_weights = base.get_weights()
        target_weights = target.get_weights()

        for i in range(len(base_weights)):
            target_weights[i] = (1. - self.tau) * target_weights[i] + self.tau * base_weights[i]
        target.set_weights(target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(cur_state)

    def get_filename(self):
        return './tf_save_ac/{}.ckpt'.format(self.env.env.spec.id)

    def save(self):
        self.saver.save(self.sess, self.get_filename())
        print("Saved")

    def restore(self):
        self.saver.restore(self.sess, self.get_filename())
        print("Restored")


def main():
    sess = tf.Session()
    K.set_session(sess)
    #env = gym.make("Pendulum-v0")
    env = gym.make("MountainCarContinuous-v0")
    actor_critic = ActorCritic(env, sess)
    #actor_critic.restore()

    num_session = 10
    for i in range(num_session):
        num_episodes = 100
        to_show = 200
        for i in range(num_episodes):
            cur_state = env.reset()
            rewards = []
            t = 0
            done = False
            while not done:
                t += 1
                if (i+1) % to_show == 0:
                    env.render()
                cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
                action = actor_critic.act(cur_state)
                action = action.reshape((1, env.action_space.shape[0]))

                new_state, reward, done, _ = env.step(action)
                new_state = new_state.reshape((1, env.observation_space.shape[0]))

                k = 20
                area = 1.0*int(k * ((new_state[0][0] - env.env.min_position) / (env.env.goal_position - env.env.min_position)))
                learning_reward = area
                rewards.append(learning_reward)
                learning_reward_array = np.array([learning_reward])

                actor_critic.remember(cur_state, action, learning_reward_array, new_state, done)

                cur_state = new_state

                if done:
                    print("Episode {} / {}".format(i+1, num_episodes))
                    print("Elapsed {} steps".format(t))
                    print('Average reward: {}'.format(np.mean(np.array(rewards))))
                    print('Max reward: {}'.format(max(rewards)))
                    print('-------------------')
            actor_critic.train(True)
        actor_critic.save()


if __name__ == "__main__":
    main()
