import numpy as np
from stable_baselines3.dqn.policies import DQNPolicy

class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, threshold=0.99, threshold_step = 1e-5, default_policy=None, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs)
        self.threshold = threshold
        self.threshold_decrement = threshold_step
        self.default_policy = default_policy

    def forward(self, obs, deterministic=False):
        # Sample a random variable
        random_var = np.random.rand()

        if random_var > self.threshold:
            # Use the custom policy to select an action
            # If no default passed in, select a random action
            if self.default_policy == None:
                action = np.random.randint(0, self.action_space.n)
            else: 
                # select an action using the default policy
                action = self.default_policy.forward(obs, deterministic)
        else:
            # Use the original DQN policy to select an action
            action = super(CustomDQNPolicy, self).forward(obs, deterministic)

        # Decrement the threshold
        self.threshold = max(0.0, self.threshold - self.threshold_decrement)
        return action
