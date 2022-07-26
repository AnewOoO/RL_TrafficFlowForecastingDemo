import gym
from gym.vector.utils import spaces
import numpy as np
MAX_ACCOUNT_BALANCE = 10000
MAX_FLOW = 700

class ITSEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(ITSEnv, self).__init__()
        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=0, high=700, shape=(1, 4), dtype=np.float32)
        self.current_step = 0
    def _next_observation(self):
        obs = np.array([
            self.df[self.current_step][0],
            self.df[self.current_step][1],
            self.df[self.current_step][2]
        ])
        return obs

    def step(self, action):
        self.current_step += 1
        if self.current_step >= 200:
            self.current_step = (self.current_step+1) % 200
        rewards = {4: -50, 3: -20, 2: -10, 1: -5, 0: 50}
        reward = rewards[abs(action - self.df[self.current_step][3])]
        done = action == self.df[self.current_step][3]
        obs = self._next_observation()
        return obs, reward, done, {}

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def render(self, mode="human"):
        print('nothing')