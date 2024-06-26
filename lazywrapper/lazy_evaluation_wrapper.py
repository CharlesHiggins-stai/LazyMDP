import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium import ActionWrapper, ObservationWrapper

class LazyEvaluationWrapper(ActionWrapper):
    """
    A wrapper which shouldn't ever be used for training of LazyModels. 
    This wraper is used to evaluate the performance of a LazyModel without the reward updating
    In essence, we can see how the policy transfers to the original MDP.
    """
    def __init__(self, env: gym.Env, default_policy, **kwargs):
        super().__init__(env)
        self.env = env
        # inherit the observation space
        self.observation_space = env.observation_space
        # inherit the action space and + 1
        self.action_space = Discrete(env.action_space.n + 1)
        self.default_policy = default_policy
        self.kwargs = kwargs
    
    def step(self, action):
        if action == self.action_space.n - 1:
            # if action is lazy, call the default policy
            obs, reward, terminated, truncated, info = self.env.step(
                self.default_policy.predict(self.env.get_last_observation())[0]
                )
            return obs, reward, terminated, truncated, info
        else:
            return self.env.step(action)
