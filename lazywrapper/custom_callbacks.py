import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class ActionProportionCallback(BaseCallback):
    def __init__(self, eval_env, verbose=0):
        super(ActionProportionCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.action_counts = None

    def _on_step(self) -> bool:
        # Initialize action counts if not already done
        if self.action_counts is None:
            action_space = self.eval_env.action_space.n
            self.action_counts = np.zeros(action_space)

        # Get the action taken in the current step
        action = self.locals['actions']
        self.action_counts[action] += 1

        return True

    def _on_rollout_end(self) -> None:
        # Calculate the proportions
        total_actions = np.sum(self.action_counts)
        action_proportions = self.action_counts / total_actions
        log_out = {}
        for action, proportion in enumerate(action_proportions):
            log_out[f'action_{action}_proportion'] = proportion
        wandb.log(log_out)

        # Reset action counts for the next rollout
        self.action_counts = None
