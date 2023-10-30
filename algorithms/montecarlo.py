import numpy as np
from algorithms.common import QTable, ReturnsTable
from helpers import TensorboardLogger

class Montecarlo_FirstVisit:
    def __init__(self, env):
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.q_table = QTable(n_states, n_actions)
        self.returns = ReturnsTable(n_states, n_actions)
        self.env = env

    def learn(self, policy, n_steps:int=100, discount_rate=1, verbose=False):
        obs = self.env.reset()
        selected_action = policy(self.q_table[obs])
        trajectory = []

        tblogger = TensorboardLogger("MonteCarloFV_(dr=" + str(discount_rate) +")" , episode_period=100)
        
        for n in range(n_steps):
            old_obs = obs
            obs, reward, done, _ = self.env.step(selected_action)

            trajectory.append({"state":old_obs, "action":selected_action, "reward":reward})
            if verbose:
                self.env.render()

            selected_action = policy(self.q_table[obs])

            if done:
                # analyze episode
                G = 0
                for i, step in enumerate(reversed(trajectory)):
                    state = step["state"]
                    action = step["action"]
                    reward = step["reward"]
                    G = G*discount_rate + reward

                    if i == 0:
                        tblogger.log(G,len(trajectory))

                    if not self.visited_state_action(state, action, trajectory, i+1): # First visit
                        if self.returns[state][action] is None:
                            self.returns[state][action] = []
                        self.returns[state][action].append(G)
                        self.q_table[state][action] = np.average(self.returns[state][action])
                if verbose:
                    print(self.q_table)
                # reset the environment and reinitialize trajectory
                if verbose:
                    print("--- EPISODE STARTS ---")
                obs = self.env.reset()
                selected_action = policy(self.q_table[obs])
                trajectory = []
                G = 0

    def visited_state_action(self, state, action, trajectory, max) -> bool:
        for step in trajectory[:-max]:
            if step["state"]== state and step["action"]==action:
                return True
        return False 
