
for i in $(seq 0  10000 50000); do
    python experiments/lazy_mdp_delayed_start.py --environment "CartPole-v1" --seed 0 --max_steps 1000000 --output_dir "experiments/lay_mdp_delayed_start_data" --env_reward_threshold 500 --penalty 0.5 --warmup_steps $i --tags "ppo" "un-LazyMDP delayed start"
    echo "CartPole-v1 done"
    python experiments/lazy_mdp_delayed_start.py --environment "LunarLander-v2" --seed 0 --max_steps 1000000 --output_dir "experiments/lazy_mdp_delayed_start_data" --env_reward_threshold 200 --penalty 0.5 --warmup_steps $i --tags "ppo" "un-LazyMDP delayed start"
    echo "LunarLander-v2 done"
done