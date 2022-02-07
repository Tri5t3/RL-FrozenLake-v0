"""
    author: Tommy Kong
    credit: Thomas Simonini
    reference: https://gist.github.com/simoninithomas/baafe42d1a665fb297ca669aa2fa6f92
"""

from math import gamma
import gym
import random
import numpy as np
import time
from collections import deque
import pickle


from collections import defaultdict


EPISODES = 20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999


def default_Q_value():
    return 0


if __name__ == "__main__":

    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)

    n_state = env.observation_space.n
    n_action = env.action_space.n

    # starts with a pessimistic estimate of zero reward for each state.
    Q_table = defaultdict(default_Q_value)
    Q_arr = np.zeros((n_state, n_action))

    episode_reward_record = deque(maxlen=100)

    for i in range(EPISODES):
        episode_reward = 0

        # TODO PERFORM Q LEARNING

        done = False
        state = env.reset()
        while not done:
            eps = random.uniform(0, 1)
            action = 0
            if eps > EPSILON:
                action = np.argmax(Q_arr[state, :])
            else:
                action = env.action_space.sample()

            s_prime, reward, done, info = env.step(action)
            episode_reward += reward

            if not done:
                Q_arr[state][action] = Q_arr[state][action] + LEARNING_RATE * \
                    (reward + DISCOUNT_FACTOR *
                     np.max(Q_arr[s_prime])
                     - Q_arr[state][action])
                # EPSILON *= EPSILON_DECAY

            if done:
                Q_arr[state][action] = Q_arr[state][action] + LEARNING_RATE * \
                    (reward - Q_arr[state][action])
                # EPSILON *= EPSILON_DECAY

            state = s_prime

            if done:
                break

        EPSILON *= EPSILON_DECAY
        episode_reward_record.append(episode_reward)

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " +
                  str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON))

    for i in range(len(Q_arr)):
        for j in range(len(Q_arr[0])):
            Q_table[(i, j)] = Q_arr[i][j]

    # print("Type of Q_table:", type(Q_table))
    # print("Size of Q_table:", len(Q_table))
    # print("Size of Q_arr:", len(Q_arr))
    # print("Size of Q_arr[0]:", len(Q_arr[0]))
    Q_arr = None

    ####DO NOT MODIFY######
    model_file = open('Q_TABLE.pkl', 'wb')
    pickle.dump([Q_table, EPSILON], model_file)
    #######################
