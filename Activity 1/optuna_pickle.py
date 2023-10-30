import optuna
import gym
import pickle
import gym
from algorithms.qlearning import QLearning
from algorithms.common import max_policy
from algorithms.common import *
import numpy as np
from optuna.samplers import TPESampler
import json
from optuna.visualization import plot_optimization_history, plot_contour, plot_slice, plot_param_importances
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Load the Q-table from pickle
with open("Activity 1\QLearning_pickle\qtable_epsilon0.0013.pickle", "rb") as f:
    q_table = pickle.load(f)

# Create the Q-Learning agent
env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True) #we tried both the determinstic and stochastic environment
algo = QLearning(env)
algo.q_table = q_table

# Define the hyperparameter space
hyperparameter_space = {
    "discount_rate": optuna.distributions.UniformDistribution(0.9, 1.0),
    "lr": optuna.distributions.LogUniformDistribution(0.001,0.01),
}


def objective(trial: optuna.Trial) -> float: #objective function that Optuna will use to optimize the hyperparameters
    discount_rate = trial.suggest_float("discount_rate", 0.9, 1.0)
    lr = trial.suggest_float("lr", 1e-5, 0.1, log=True)

    q_learning_agent = QLearning(env)
    q_learning_agent.q_table = q_table
    epsilon=0.1
    epsilon_greedy_policy = EpsilonGreedyPolicy(epsilon=epsilon)
    q_learning_agent.learn(epsilon_greedy_policy, n_steps=10000, lr=lr)

    avg_reward, avg_steps = evaluate_policy(q_learning_agent.env, q_learning_agent.q_table, max_policy, n_episodes=100, verbose=False)

    return avg_reward

def main():
    # Clear previous logs
    os.system("rm -rf ./logs")
    seed = 3
    np.random.seed(seed)

    # Initialize the sampler and create the study
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(study_name="FrozenLake8x8", direction="maximize", sampler=sampler)

    # Perform optimization
    study.optimize(objective, n_trials=50)

    # Visualize the results
    fig = plot_optimization_history(study)
    fig.show()

# Plot the parameter contour
    fig = plot_contour(study)
    fig.show()

# Plot the parameter slice
    fig = plot_slice(study)
    fig.show()

# Plot the parameter importances
    fig = plot_param_importances(study)
    fig.show()

    # Print the result on the screen
    best_trial = study.best_trial
    print("Best trial:")
    print("  Value: ", best_trial.value)
    print("  Params: ")
    best_trial_params = json.dumps(best_trial.params, sort_keys=True, indent=4)
    print(best_trial_params)

if __name__ == "__main__":
    main()
#best trial for the stochastic environnment
'''
Best trial:
  Value:  0.71
  Params:
{
    "discount_rate": 0.9961459723114788,
    "lr": 0.06698053340104898
}'''