python train.py --device 1 --alg dqn --use_mf True
python evaluate.py --device 1 --alg ppo --agent-num 150 --use_mf True --init-checkpoint 2000000 --checkpoint-dir checkpoint
python evaluate_adv.py --device 1 --alg ppo --agent-num 50 --checkpoint-dir checkpoint --init-checkpoint 2000000 --adv-checkpoint-dir checkpoint --adv-checkpoint 1800000
python train_adv.py --device 1 --alg ppo --init-checkpoint 2000000 --checkpoint-dir checkpoint
python train_adv.py --device 1 --alg ppo --use_mf True --init-checkpoint 2000000 --checkpoint-dir checkpoint
python train_Q.py --device 1 --alg ppo --agent-num 50 --init-checkpoint 2000000 --checkpoint-dir checkpoint
python train_V.py --agent-num 50 --init-checkpoint 2000000 --qfunc-checkpoint 2000000 --checkpoint-dir checkpoint
python train_adv.py --agent-num 50 --adv-num 4 --adv-method ours --init-checkpoint 2000000 --vfunc-checkpoint 648 --checkpoint-dir checkpoint