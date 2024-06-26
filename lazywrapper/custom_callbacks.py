import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
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


class OriginalEvalLogger(BaseCallback):
    def __init__(self, eval_env, n_eval_episodes=10, eval_freq=1000, verbose=1, log_path = "./eval_logs"):
        super(OriginalEvalLogger, self).__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_step_counter = 0

    def _on_step(self) -> bool:
        # Increment step counter
        self.eval_step_counter += 1
        
        # Check if it's time to perform evaluation
        if self.eval_step_counter % self.eval_freq == 0:
            self.perform_evaluation()
        
        return True

    def perform_evaluation(self):
        # Perform evaluation using the environment
        # eval_results = self.model.evaluate(self.eval_env, n_eval_episodes=self.n_eval_episodes, return_episode_rewards=True)
        rewards, lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=False,
                deterministic=True,
                return_episode_rewards=True
            )
        # Calculate mean reward
        mean_reward, std_reward = np.mean(rewards), np.std(rewards)
        mean_ep_length, std_ep_length = np.mean(lengths), np.std(lengths)
        # Log to WandB
        wandb.log({'eval/original_mean_reward': mean_reward, 'eval/original_ep_length': mean_ep_length}, step=self.num_timesteps)

    def _on_rollout_end(self):
        super(OriginalEvalLogger, self)._on_rollout_end()
        # Optionally, force an evaluation at the end of each rollout
        # self.perform_evaluation()
