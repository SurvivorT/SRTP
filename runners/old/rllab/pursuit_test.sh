python run_pursuit.py --log_dir gru_test --n_iter 500 --n_evaders 5 --n_pursuers 4 --control decentralized --n_timesteps 10000 --obs_range 5 --map_file ../maps/map_pool32.npy --policy_hidden_sizes 32 --flatten --recurrent --sampler_workers 1 --surround --discount 0.99 --max_kl 0.1
