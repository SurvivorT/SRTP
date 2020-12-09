python3 runners/run_multiwalker.py rllab \ # Use rllab for training
    --control decentralized \ # Decentralized training protocol
    --policy_hidden 100,50,25 \ # Set MLP policy hidden layer sizes
    --n_iter 200 \ # Number of iterations
    --n_walkers 2 \ # Starting number of walkers
    --batch_size 24000 \ # Number of rollout waypoints
    --curriculum lessons/multiwalker/env.yaml