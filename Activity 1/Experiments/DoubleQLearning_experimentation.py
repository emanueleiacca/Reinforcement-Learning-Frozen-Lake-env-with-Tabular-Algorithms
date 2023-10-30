import os
import gym
import random
import time
import numpy as np
import pickle
from helpers import FrozenLakeDenseRewardsWrapper
from algorithms.doubleqlearning import DoubleQLearning
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

#setting up a function that train and evaluate the DoubleQLearning agent
def train_and_evaluate_algorithm(algo, env, n_steps, tb_writer, epsilon=1.0):
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    
    start_time = time.time()
    algo.learn(epsilon_greedy_policy, n_steps)
    elapsed_time = time.time() - start_time
    print(f"Training time: {elapsed_time:.4f} secs")
    q_table1 = algo.q_table1
    q_table2 = algo.q_table2
    evaluate_policy(env, q_table1.table +q_table2.table, max_policy, n_episodes=100, verbose=False)

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
        qtable_filename = f"qtable_epsilon_2ql{epsilon:.4f}.pickle"

        algo = DoubleQLearning(env)

        print(f"Testing DoubleQLearning with Epsilon={epsilon:.4f}")
        train_and_evaluate_algorithm(algo, env, n_steps, tb_writer, epsilon)

        q_table1 = algo.q_table1
        q_table2 = algo.q_table2
        save_q_table(q_table1, qtable_filename)
        save_q_table(q_table2,  "second-" + qtable_filename)
