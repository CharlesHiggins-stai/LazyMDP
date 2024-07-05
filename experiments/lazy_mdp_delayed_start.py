# set the path to the project directory
import sys
sys.path.append('/home/tromero_client/LazyMDP')
# training imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from baselines.random_policy import RandomPolicy
from wandb.integration.sb3 import WandbCallback
import wandb
import gymnasium as gym
# custom imports
from lazywrapper import LazyWrapperDelayedStart, LastObservationWrapper, LazyEvaluationWrapper
from lazywrapper.custom_callbacks import ActionProportionCallback, OriginalEvalLogger
# general imports
import argparse 
import os
import copy

def make_custom_env(original_env, loaded_default_policy, penalty, warmup_steps):
    def _init():
        env = LazyWrapperDelayedStart(LastObservationWrapper(gym.make(original_env)), default_policy=loaded_default_policy, penalty = penalty, warmup_steps=warmup_steps)
        return env
    return _init

def make_custom_eval_env(original_env, loaded_default_policy):
    def _init():
        env = LazyEvaluationWrapper(LastObservationWrapper(gym.make(original_env)), default_policy=loaded_default_policy)
        return env
    return _init

def get_default_policy(env:str = None, policy_type:str='suboptimal'):
    """ 
    Cheap and dirty function meant to be easy to read because it caused Too. Many. Fucking. Mistakes. So we're doing it the dumb way. 
    Iterates through the various forms of policies, and builds the default policy  
    """
    
    if policy_type not in ['optimal', "suboptimal", 'random']:
        raise ValueError(f"default policy type {policy_type} not supported. Must be one of: ['optimal', suboptimal', 'random']")
    elif policy_type == "optimal":
        file_path = f"baselines/optimal_pretrained_policies/ppo_" + env + "_0"
        try:
            policy = PPO.load(file_path)
        except:
            raise ValueError(f"Could not load policy from file path: {file_path}")
    elif policy_type == "suboptimal":
        file_path = f"baselines/suboptimal_pretrained_policies/ppo_" + env + "_0"
        try:
            policy = PPO.load(file_path)
        except:
            raise ValueError(f"Could not load policy from file path: {file_path}")
    elif policy_type == "random":
        temp_env = gym.make(env)
        action_space = copy.copy(temp_env.action_space)
        del temp_env
        policy = RandomPolicy(action_space=action_space)
    else:
        raise ValueError(f"Something went wrong here")
    print("We are working with a {policy_type} default policy")
    return policy


def train_lazy_mdp(
    environment:str, 
    seed:int = 0, 
    output_dir:str = "", 
    total_steps:int = 25000, 
    wandb_project_name = "LazyMDP", 
    env_reward_threshold: int = 50,
    penalty: float = 1.0,
    tags = ["baseline", "ppo"],
    **kwargs):
    """Train a LazyMDP policy for a given environment.

    Args:
        environment (str): generic gym environment name
        seed (int, optional): RandomSeed to be set. Defaults to 0.
        output_dir (str, optional): Filepath for results and logging. Defaults to "".
        total_steps (int, optional): Total steps depending on environment. Defaults to 25000.
        wandb_project_name (str, optional): WandB project name. Defaults to "LazyMDP".
        env_reward_threshold (int, optional): Threshold at which training is complete --- game has been won. Defaults to 50.
        penalty (int, optional): Penalty for selecting the lazy action. Defaults to -1.
    """

    
    default_policy = get_default_policy(env=environment, policy_type=wandb.config.default_policy_type)
    # Set up Parallel environments -- vec env for trainig, single for evaluation
    vec_env = make_vec_env(env_id=make_custom_env(original_env=environment, loaded_default_policy=default_policy, penalty=penalty, warmup_steps=wandb.config.warmup_steps), n_envs=4)
    eval_env = make_vec_env(env_id=make_custom_eval_env(environment, default_policy), n_envs=1)
    # Set up Callbacks for evaluation
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=env_reward_threshold, verbose=1)
    eval_callback_training = EvalCallback(vec_env, callback_on_new_best=callback_on_best, n_eval_episodes=10, eval_freq=1000, verbose=1)
    eval_callback_basic = OriginalEvalLogger(eval_env, n_eval_episodes=10, eval_freq=1000, verbose=1, log_path = f"{output_dir}/eval_logs")

    # set up callback to record action proportions during evaluation
    proportion_callback = ActionProportionCallback(vec_env, verbose=1)
    # setup wandb callback
    wandb_callback = WandbCallback(verbose=2)
    # Set up model 
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=f"{output_dir}/tensorboard")
    # run model
    model.learn(total_timesteps=total_steps, callback=[eval_callback_training, eval_callback_basic, proportion_callback, wandb_callback])
    model.save(f"{output_dir}/ppo_{environment}_{penalty}_{seed}")

    # obs = vec_env.reset()
    # while True:
    #     action, _states = model.predict(obs)
    #     obs, _, _, _ = vec_env.step(action)
    #     vec_env.render("human")
        
    
if __name__ == "__main__":
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description='Process input arguments for a simulation.')

    # Adding arguments
    parser.add_argument('--environment', type=str, help='Name of the environment to simulate.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0).')
    parser.add_argument('--max_steps', type=int, default= 25000, help='Maximum number of steps to simulate.')
    parser.add_argument('--output_dir', type=str, default = "experiments/data", help='Directory to save output results.')
    parser.add_argument('--env_reward_threshold', type=int, default = 50, help='Reward threshold to stop training.')
    parser.add_argument('--penalty', type=float, default = -1, help='Penalty for selecting the lazy action.')
    parser.add_argument('--default_policy_type', type=str, default="suboptimal", help= "The defining characteristic of the default policy")
    parser.add_argument('--warmup_steps', type=int, default = 1000, help='Number of steps to warm up before using the default policy.')
    parser.add_argument('--tags', nargs='+', default = ["experiment", "ppo"], help='Tags for wandb runs')

    # Parse the arguments
    args = parser.parse_args()
    wandb.init(
        project="LazyMDP", 
        sync_tensorboard=True,
        tags = [*args.tags, args.environment]
        )
    # set wandb config
    wandb.config.update(args)
    extra_config = {
        "experiment_class": "unLazyMDP Delayed Start"
        }
    wandb.config.update(extra_config)
    
    # Print the inputs (You can replace this section with the actual logic)
    print(f"Running simulation in environment: {args.environment}")
    print(f"Random seed: {args.seed}")
    print(f"Maximum steps: {args.max_steps}")
    print(f"Output will be saved in: {args.output_dir}")

    train_lazy_mdp(
        environment = args.environment, 
        seed = args.seed, 
        output_dir = args.output_dir, 
        total_steps = args.max_steps, 
        env_reward_threshold=args.env_reward_threshold,
        penalty=args.penalty,
        tags = args.tags
    )
    print("Done")