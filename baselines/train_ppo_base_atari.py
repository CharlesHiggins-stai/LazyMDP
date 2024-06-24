# training imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack 
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from wandb.integration.sb3 import WandbCallback
import wandb
import gymnasium as gym
# general imports
import argparse
import os


def train_default_policy(
    environment:str = "BreakoutNoFrameskip-v4", 
    seed:int = 0, 
    output_dir:str = "", 
    total_steps:int = 10000000, 
    wandb_project_name = "LazyMDP", 
    env_reward_threshold: int = 400, 
    tags = ["baseline", "ppo", "atari"],
    **kwargs):
    """Train a default PPO policy on a given environment

    Args:
        environment (str): environment string (generic gym environment name)
        seed (int, optional): random seed. Defaults to 0.
        output_dir (str, optional): _description_. Defaults to "".
        total_steps (int, optional): _description_. Defaults to 25000.
    """

    # def make_internal_atari_env(original_env):
    #     def _init():
    #         env = AtariWrapper(gym.make(original_env))
    #         return env
    #     return _init
    
    # Set up Parallel environments -- vec env for trainig, single for evaluation
    vec_env = make_atari_env(environment, n_envs=8, seed=seed)  
    vec_env = VecFrameStack(vec_env, n_stack=4)
 
    # Set up Callbacks for evaluation
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=env_reward_threshold, verbose=1)
    eval_callback = EvalCallback(vec_env, callback_on_new_best=callback_on_best, n_eval_episodes=10, eval_freq=1000, verbose=1)
    wandb_callback = WandbCallback(verbose=2)
    # Set up model 
    model = PPO(
        "CnnPolicy", 
        vec_env, 
        verbose=1,
        tensorboard_log=f"{output_dir}/tensorboard", 
         batch_size=256, 
        clip_range=0.1, 
        ent_coef=0.01, 
        learning_rate=2.5e-4, 
        gamma=0.99, 
        lambda_=0.95, 
        n_steps=128, 
        n_epochs=4
        )
    # run model
    model.learn(total_timesteps=total_steps, callback=[eval_callback, wandb_callback])
    model.save(f"{output_dir}/ppo_{environment}_{seed}")

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
    parser.add_argument('--output_dir', type=str, default = "baselines/pretrained_policies", help='Directory to save output results.')
    parser.add_argument('--env_reward_threshold', type=int, default = 50, help='Reward threshold to stop training.')
    parser.add_argument('--tags', nargs='+', default = ["baseline", "ppo"], help='Tags for wandb runs')
    # Parse the arguments
    args = parser.parse_args()

    # Print the inputs (You can replace this section with the actual logic)
    print(f"Running simulation in environment: {args.environment}")
    print(f"Random seed: {args.seed}")
    print(f"Maximum steps: {args.max_steps}")
    print(f"Output will be saved in: {args.output_dir}")

    wandb.init(
        project="LazyMDP", 
        sync_tensorboard=True,
        tags = [*args.tags, args.environment]
        )
    
    wandb.config.update(args)
    train_default_policy(
        environment = args.environment, 
        seed = args.seed, 
        output_dir = args.output_dir, 
        total_steps = args.max_steps, 
        env_reward_threshold=args.env_reward_threshold
    )
    print("Done")
    wandb.finish()