# TAXI

### 1. Training without adversarial agents

```Bash
python train.py --alg ppo --seed 0 --agent-num 50
```

- `--alg`: Specifies that the PPO algorithm is used for training.
- `--seed`: Sets the random seed for reproducibility, in this case, seed value is set to 0.
- `--agent-num`: Indicates the number of agents to be trained.

### 2. Train the Q network

```Bash
python train_Q.py --alg ppo --seed 0 --agent-num 50 --init-checkpoint 2000000 --checkpoint-dir checkpoints/50
```

- `--init-checkpoint`: Loads the checkpoint file from step 1, the file name must be `policy_<init-checkpoint>.pth`.
- `--checkpoint-dir`: Specifies the directory where model checkpoints are saved during training.

### 3. Train the V network

```Bash
python train_V.py --alg ppo --seed 0 --agent-num 50 --adv-num 16 --init-checkpoint 2000000 --qfunc-checkpoint 2000000 --checkpoint-dir checkpoints/50
```

- `--adv-num`: Specifies the number of adversarial agents.
- `--init-checkpoint`: Loads the checkpoint file from step 1, the file name must be `policy_<init-checkpoint>.pth`.
- `--qfunc-checkpoint`: Loads the checkpoint file of the Q network from step 2, the file name must be `Qfunc_<qfunc-checkpoint>.pth`.
- `--checkpoint-dir`: Specifies the directory where the model checkpoints will be saved. Both `policy_<init-checkpoint>.pth` and `Qfunc_<qfunc-checkpoint>.pth` should be in this directory.
