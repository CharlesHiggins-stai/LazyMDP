# first generate the baselines
python baselines/train_ppo_base.py --environment "CartPole-v1" --seed 0 --max_steps 1000000 --output_dir "baselines/suboptimal_pretrained_policies" --env_reward_threshold 400 --tags "suboptimal" "ppo"
echo "CartPole-v1 done" 
python baselines/train_ppo_base.py --environment "LunarLander-v2" --seed 0 --max_steps 1000000 --output_dir "baselines/suboptimal_pretrained_policies" --env_reward_threshold 100 --tags "suboptimal" "ppo"
echo "LunarLander-v2 done"

# now train plain vanilla PPO to convergence
python baselines/train_ppo_base.py --environment "CartPole-v1" --seed 0 --max_steps 1000000 --output_dir "baselines/optimal_pretrained_policies" --env_reward_threshold 500 --tags "optimal" "ppo" "baseline"
echo "CartPole-v1 done"
python baselines/train_ppo_base.py --environment "LunarLander-v2" --seed 0 --max_steps 1000000 --output_dir "baselines/optimal_pretrained_policies" --env_reward_threshold 200 --tags "optimal" "ppo" "baseline"
echo "LunarLander-v2 done"

# now train lazy MDP original
python experiments/lazy_mdp_original.py --environment "CartPole-v1" --seed 0 --max_steps 1000000 --output_dir "experiments/lazy_mdp_original_data" --env_reward_threshold 500 --penalty 0.99 --tags "ppo" "original LazyMDP"
echo "Original Lazy MDP CartPole-v1 done"
python experiments/lazy_mdp_original.py --environment "LunarLander-v2" --seed 0 --max_steps 1000000 --output_dir "experiments/lazy_mdp_original_data" --env_reward_threshold 200 --penalty 0.99 --tags "ppo" "original LazyMDP"
echo "Original LunarLander-v2 done"

# now train un-LazyMDP
python experiments/lazy_mdp.py --environment "CartPole-v1" --seed 0 --max_steps 1000000 --output_dir "experiments/lay_mdp_data" --env_reward_threshold 500 --penalty 0.99 --tags "ppo" "un-LazyMDP"
echo "CartPole-v1 done"
python experiments/lazy_mdp.py --environment "LunarLander-v2" --seed 0 --max_steps 1000000 --output_dir "experiments/lazy_mdp_data" --env_reward_threshold 200 --penalty 0.99 --tags "ppo" "un-LazyMDP"
echo "LunarLander-v2 done"

# now train Antonioni PPO