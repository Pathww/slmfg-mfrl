# CUDA_VISIBLE_DEVICES=0 python train.py --env-name taxi --alg ppo --map-N 10 --map-M 10 --agent-num 10 &
# CUDA_VISIBLE_DEVICES=0 python train.py --env-name taxi --alg ppo --map-N 20 --map-M 20 --agent-num 40 &
python train.py --device 1 --env-name taxi --alg dqn --map-M 10 --map-N 10 --agent-num 10