for sd in $(seq 0 1 5); do
    for policy_type in suboptimal random optimal 
    do
        python experiments/lazy_mdp_delayed_start.py --environment "CartPole-v1" --seed $sd --max_steps 1000000 --output_dir "experiments/lazy_mdp_delayed_start_data" --env_reward_threshold 500 --penalty 0.1 --warmup_steps 0 --default_policy_type $policy_type --tags "ppo" "un-LazyMDP delayed start"
        echo "CartPole-v1 done"

        python experiments/lazy_mdp_delayed_start.py --environment "LunarLander-v2" --seed $sd --max_steps 1000000 --output_dir "experiments/lazy_mdp_delayed_start_data" --env_reward_threshold 200 --penalty 0.1 --warmup_steps 0 --default_policy_type $policy_type --tags "ppo" "un-LazyMDP delayed start"
        echo "LunarLander-v2 done"

        python experiments/lazy_mdp_delayed_start.py --environment "CartPole-v1" --seed $sd --max_steps 1000000 --output_dir "experiments/lazy_mdp_delayed_start_data" --env_reward_threshold 500 --penalty 0.1 --warmup_steps 50000 --default_policy_type $policy_type --tags "ppo" "un-LazyMDP delayed start"
        echo "CartPole-v1 done"

        python experiments/lazy_mdp_delayed_start.py --environment "LunarLander-v2" --seed $sd --max_steps 1000000 --output_dir "experiments/lazy_mdp_delayed_start_data" --env_reward_threshold 200 --penalty 0.1 --warmup_steps 50000 --default_policy_type $policy_type --tags "ppo" "un-LazyMDP delayed start"
        echo "LunarLander-v2 done"
    done
done