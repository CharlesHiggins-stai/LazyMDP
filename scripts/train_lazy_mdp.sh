# define the fixed variables
for sd in $(seq 1 1 5); do 
    for i in $(seq 0 0.1 0.9); do 
        python experiments/lazy_mdp.py --environment "CartPole-v1" --seed $sd --max_steps 1000000 --output_dir "experiments/lazy_mdp_data" --env_reward_threshold 500 --penalty $i --tags "ppo" "un-LazyMDP"
        echo "CartPole-v1 done"
        python experiments/lazy_mdp.py --environment "LunarLander-v2" --seed $sd --max_steps 1000000 --output_dir "experiments/lazy_mdp_data" --env_reward_threshold 200 --penalty $i --tags "ppo" "un-LazyMDP"
        echo "LunarLander-v2 done"
    done
done