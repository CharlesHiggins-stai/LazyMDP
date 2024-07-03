# A wrapper which allows for any gym environment to be modified to be lazy
import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium import ActionWrapper, ObservationWrapper


class LazyWrapperOriginal(ActionWrapper):
    def __init__(self, env: gym.Env, default_policy, penalty, **kwargs):
        super().__init__(env)
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
            # if action is lazy, call the default policy
            obs, reward, terminated, truncated, info = self.env.step(
                self.default_policy.predict(self.env.get_last_observation())[0]
                )
            return obs, reward, terminated, truncated, info
        else:
            obs, reward, terminated, truncated, info = self.env.step(action)
            return obs, reward - self.penalty, terminated, truncated, info


