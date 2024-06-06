python baselines/train_ppo_base.py --environment "CartPole-v1" --seed 0 --max_steps 1000000 --output_dir "baselines/suboptimal_pretrained_policies" --env_reward_threshold 400 --tags "suboptimal" "ppo"
echo "CartPole-v1 done" 
python baselines/train_ppo_base.py --environment "LunarLander-v2" --seed 0 --max_steps 1000000 --output_dir "baselines/suboptimal_pretrained_policies" --env_reward_threshold 100 --tags "suboptimal" "ppo"
echo "LunarLander-v2 done"
# python baselines/train_ppo_base.py --environment "FrozenLake-v1" --seed 0 --max_steps 500000 --output_dir "baselines/pretrained_policies" --env_reward_threshold 200
# echo "FrozenLake-v1 done"