# define the fixed variables
python experiments/lazy_mdp_atari.py --environment "ALE/B" --seed 0 --max_steps 250000 --output_dir "experiments/pretrained_policies" --env_reward_threshold 400 --penalty 0.5
echo "Breakout-v5 done"
