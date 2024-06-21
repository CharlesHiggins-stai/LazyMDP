# define the fixed variables
output_dir="experiments/pretrained_policies"
max_steps=1000000
seed=0
penalty=0.99

python experiments/lazy_mdp.py --environment "CartPole-v1" --seed 0 --max_steps 1000000 --output_dir "experiments/pretrained_policies" --env_reward_threshold 500 --penalty 0.99
echo "CartPole-v1 done"
python experiments/lazy_mdp.py --environment "LunarLander-v2" --seed 0 --max_steps 1000000 --output_dir "experiments/pretrained_policies" --env_reward_threshold 200 --penalty 0.99
echo "LunarLander-v2 done"