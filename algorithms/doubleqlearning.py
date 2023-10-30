'''
3 qtables
teaching it with the first and second qtables

num = random.randint(0,1) -> 50% choose from 1. or 2. qtable

It has all we need:
    selected_action = policy(self.q_table[obs])
    # Q-Learning's max-Q value instead of the expected value. 
    max_q_value = max(self.q_table[obs]) #this is the only part we change
    td = reward + discount_rate * max_q_value
    self.q_table[previous_obs][previous_action] += lr*(td-self.q_table[previous_obs][previous_action])

We only have to write 5-10 lines to the original qlearning code
'''

import random
import numpy as np
import gym 
from algorithms.common import *
from helpers import TensorboardLogger

#we need to change the learn method
class DoubleQLearning:
    def __init__(self, env):
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        # using qtables
        # self.q_table = QTable(n_states, n_actions)
        self.q_table1 = QTable(n_states, n_actions)
        self.q_table2 = QTable(n_states, n_actions)
        self.env = env

    def learn(self, policy, n_steps:int=100, discount_rate=1, lr=0.01, lrdecay=1.0, n_episodes_decay=100, tb_episode_period=100, verbose=False):
        obs = self.env.reset()
        selected_action= policy(self.q_table1[obs])
        tblogger = TensorboardLogger("QLearning_(dr=" + str(discount_rate) +"-lr=" + str(lr) + "-lrdecay=" + str(lrdecay) + "e"+ str(n_episodes_decay) + ")" , episode_period=tb_episode_period)

        n_episodes = 0
        episode_reward = 0
        episode_steps = 0

        for n in range(n_steps):
            previous_obs = obs
            previous_action = selected_action
            obs, reward, done, _ = self.env.step(selected_action)

            episode_reward += reward
            episode_steps += 1

            if verbose:
                self.env.render()

            # random generator for the choice -> 50% choose from 1. or 2. qtable
            num = random.randint(0,1)

            if num == 0: 
                selected_action= policy(self.q_table1[obs])
                max_q_value1 = max(self.q_table1[obs])
                td = reward + discount_rate * max_q_value1
                self.q_table2[previous_obs][previous_action] += lr*(td-self.q_table2[previous_obs][previous_action])
            else:
                selected_action= policy(self.q_table2[obs])
                max_q_value2 = max(self.q_table2[obs]) 
                td = reward + discount_rate * max_q_value2
                self.q_table1[previous_obs][previous_action] += lr*(td-self.q_table1[previous_obs][previous_action])

            lr = calculate_lr_decay(lr=lr, decay_rate=lrdecay, n_steps=n)

            if done:
                tblogger.log(episode_reward, episode_steps)
                # reset the environment and reinitialize trajectory
                if verbose:
                    print("--- EPISODE STARTS ---")
                episode_reward = 0
                episode_steps = 0
                obs = self.env.reset()

                # lrdacay update
                n_episodes += 1
                if n_episodes % n_episodes_decay == 0:
                    lr *= lrdecay

    def calculate_average_qvalue(self, values, epsilon=0):
        max_value = max (values)
        n_actions = len(values)
        n_greedy_actions = 0
        for v in values:
            if v == max_value:
                n_greedy_actions += 1

        non_greedy_action_probability = epsilon / n_actions
        greedy_action_probability = ((1 - epsilon) / n_greedy_actions) + non_greedy_action_probability

        result = 0
        for v in values:
            if v == max_value:
                result += v * greedy_action_probability
            else:
                result += v * non_greedy_action_probability

        return result