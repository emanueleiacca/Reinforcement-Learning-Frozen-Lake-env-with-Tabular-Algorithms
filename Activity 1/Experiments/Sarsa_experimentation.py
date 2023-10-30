import os
import gym
import random
import time
import numpy as np
import pickle
from helpers import FrozenLakeDenseRewardsWrapper
from algorithms.sarsa import Sarsa
from algorithms.common import EpsilonGreedyPolicy, evaluate_policy, max_policy
from algorithms.my_function import custom_epsilon
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt 

#setting up the Frozen Lake environnement
def setup_environment(seed):
    env = FrozenLakeDenseRewardsWrapper(gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True))
    random.seed(seed)
    env.seed(seed)
    return env

#setting up a function that train and evaluate the SARSA agent
def train_and_evaluate_algorithm(algo, env, n_steps, tb_writer, epsilon=1.0):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps)
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.4f} secs")

    evaluate_policy(env, algo.q_table, max_policy, n_episodes=100, verbose=False)
#saving the Q-table to a pickle file
def save_q_table(q_table, filename):
    with open(filename, "wb") as f:
        pickle.dump(q_table, f)

if __name__ == "__main__":
    tb_writer = SummaryWriter(log_dir=r'Activity 1\logs')

    seed = 3
    n_steps = 3_000_000
    env = setup_environment(seed)
    total_steps = 30

    max_steps = 512
    gamma = 0.95

    for step in range(total_steps):
        epsilon = custom_epsilon(step, total_steps)
        qtable_filename = f"qtable_epsilon_Sarsa{epsilon:.4f}.pickle"

        algo = Sarsa(env) 

        print(f"Testing Sarsa with Epsilon={epsilon:.4f}")
        train_and_evaluate_algorithm(algo, env, n_steps, tb_writer, epsilon)

        q_table = algo.q_table
        save_q_table(q_table, qtable_filename)

