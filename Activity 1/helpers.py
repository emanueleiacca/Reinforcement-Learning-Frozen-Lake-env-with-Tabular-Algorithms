from collections import deque
import datetime
from statistics import mean
import gym
import numpy as np
from gym.spaces import Box, Discrete
from torch.utils.tensorboard.writer import SummaryWriter

class FrozenLakeDenseRewardsWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        # if hole then reward -1
        if reward == 0 and done:
            reward = -1
        return next_state, reward, done, info


class TensorboardLogger():
    def __init__(self, algo_name="algo", episode_period=1, log_dir="logs"):
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(self.log_dir + "/" + algo_name + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.episode_rewards = deque([], maxlen=episode_period)
        self.episode_steps = deque([], maxlen=episode_period)
        self.total_steps = 0
        
    def log(self, episode_reward, episode_steps):
        self.total_steps += episode_steps

        self.episode_rewards.append(episode_reward)
        self.episode_steps.append(episode_steps)
        average_rewards = mean(self.episode_rewards)
        average_steps = mean(self.episode_steps)


        self.tb_writer.add_scalar("train/reward", average_rewards, self.total_steps)
        self.tb_writer.add_scalar("train/episode_len", average_steps, self.total_steps) 



# Wrapper created by lilianweng
# https://github.com/lilianweng/deep-reinforcement-learning-gym/blob/master/playground/utils/wrappers.py
class DiscretizedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_bins=10, low=None, high=None):
        super().__init__(env)
        assert isinstance(env.observation_space, Box)

        low = self.observation_space.low if low is None else low
        high = self.observation_space.high if high is None else high

        low = np.array(low)
        high = np.array(high)

        self.n_bins = n_bins
        self.val_bins = [np.linspace(l, h, n_bins + 1) for l, h in
                         zip(low.flatten(), high.flatten())]
        self.ob_shape = self.observation_space.shape

        print("New observation space:", Discrete((n_bins + 1) ** len(low)))
        self.observation_space = Discrete(n_bins ** len(low))

    def _convert_to_one_number(self, digits):
        return sum([d * ((self.n_bins + 1) ** i) for i, d in enumerate(digits)])

    def observation(self, observation):
        digits = [np.digitize([x], bins)[0]
                  for x, bins in zip(observation.flatten(), self.val_bins)]
        return self._convert_to_one_number(digits)
