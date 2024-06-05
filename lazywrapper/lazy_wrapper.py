# A wrapper which allows for any gym environment to be modified to be lazy
import gymnasium as gym
from gymnasium.spaces import Discrete


class LazyWrapper(gym.Env):
    def __init__(self, env: gym.Env, default_policy, penalty, **kwargs):
        super().__init__()
        # save environment
        self.env = env
        # inherit the observation space
        self.observation_space = env.observation_space
        # inherit the action space and + 1
        self.action_space = Discrete(env.action_space.n + 1)
        self.default_policy = default_policy
        self.penalty = penalty
        self.kwargs = kwargs
    
    def step(self, action):
        if action == self.action_space.n - 1:
            obs, reward, terminated, truncated, info, done = self.default_policy(self.env, **self.kwargs)
            return obs, reward + self.penalty, terminated, truncated, info, done
        else:
            return self.env.step(action)
