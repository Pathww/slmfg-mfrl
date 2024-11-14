python train.py --device 1 --alg dqn --use_mf True
python evaluate.py --device 1 --alg ppo --agent-num 150 --use_mf True --init-checkpoint 2000000 --checkpoint-dir ppo-render/e-150