import pickle
import gym
from algorithms.qlearning import QLearning
from algorithms.common import max_policy
from algorithms.common import *

#loading the qtable from Pickle
with open("Activity 1\QLearning_pickle\qtable_epsilon0.0013.pickle", "rb") as f:
    q_table = pickle.load(f)
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
algo = QLearning(env)  
algo.q_table = q_table
evaluate_policy(env, algo.q_table, max_policy, n_episodes=100, verbose=False)
