# training imports
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from wandb.integration.sb3 import WandbCallback
import wandb
import gymnasium as gym
# general imports
import argparse
import os


def train_default_policy(
    environment:str, 
    seed:int = 0, 
    output_dir:str = "", 
    total_steps:int = 25000, 
    wandb_project_name = "LazyMDP", 
    env_reward_threshold: int = 50, 
    tags = ["baseline", "ppo"],
    **kwargs):
    """Train a default PPO policy on a given environment

    Args:
        environment (str): environment string (generic gym environment name)
        seed (int, optional): random seed. Defaults to 0.
        output_dir (str, optional): _description_. Defaults to "".
        total_steps (int, optional): _description_. Defaults to 25000.
    """
    # set wandb config
    config = {
        "environment": environment, 
        # "seed": seed,
        "total_steps": total_steps
        }
    
    wandb.init(
        project=wandb_project_name, 
        sync_tensorboard=True,
        tags = [*tags, environment],
        config=config
        )
    
    # Set up Parallel environments -- vec env for trainig, single for evaluation
    vec_env = make_vec_env(environment, n_envs=4)
    eval_env = make_vec_env(environment)
    # Set up Callbacks for evaluation
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=env_reward_threshold, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, n_eval_episodes=10, eval_freq=1000, verbose=1)
    wandb_callback = WandbCallback(verbose=2)
    # Set up model 
    model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=f"{output_dir}/tensorboard")
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

    train_default_policy(
        environment = args.environment, 
        seed = args.seed, 
        output_dir = args.output_dir, 
        total_steps = args.max_steps, 
        env_reward_threshold=args.env_reward_threshold
    )
    print("Done")