from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import argparse
import torch
import os
from agent.SLMFG import SLMFG
import setproctitle

parser = argparse.ArgumentParser()
# parser.add_argument('--exp_name', type=str, default='aaa', help='name')
parser.add_argument('--use_mf', type=bool, default=False, help='mean field method')
parser.add_argument('--render', type=bool, default=False, help='render')
parser.add_argument('--render-every', type=int, default=5, help='render every')
parser.add_argument('--agent-num', type=int, default=50, help='Number of agents')
parser.add_argument('--adv', type=bool, default=False, help='train adv')

parser.add_argument('--seed', type=int, default=1113, help='Random seed')
parser.add_argument('--map-str', type=str, default='grid', help='Map')
parser.add_argument('--map-M', type=int, default=10, help='M for grid map')
parser.add_argument('--map-N', type=int, default=10, help='N for grid map')
parser.add_argument('--order-num', type=int, default=100, help='Number of orders')
parser.add_argument('--order-dist', type=str, default='gaussian', help='uniform, gaussian')
parser.add_argument('--order-price-min', type=int, default=1, help='Min price of an order')
parser.add_argument('--order-price-max', type=int, default=2, help='Max price of an order')
parser.add_argument('--reward-method', type=str, default='log', help='avg, log, log2')
parser.add_argument('--speed', type=int, default=5, help='Speed for agents')
parser.add_argument('--episode-len', type=int, default=20, help='Maximum episode length')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
parser.add_argument('--lr', type=float, default=0.00003, help='Learning rate for DQN')
parser.add_argument('--lr-a', type=float, default=0.00003, help='Learning rate for actor')
parser.add_argument('--lr-c', type=float, default=0.0003, help='Learning rate for critic')
parser.add_argument('--record', type=str, default='record', help='Directory record')
parser.add_argument('--checkpoint-episodes', type=int, default=200000, help='Frequency of saving checkpoints')
parser.add_argument('--max-episodes', type=int, default=2000000, help='Maximum episodes')
parser.add_argument('--env-name', type=str, default='taxi', help='Experiment name')
parser.add_argument('--init-checkpoint', type=int, default=0, help='Pretrained checkpoint')
parser.add_argument('--min-agent-num', type=int, default=20, help='Number of agents')
parser.add_argument('--max-agent-num', type=int, default=20, help='Number of agents')
parser.add_argument('--local-obs', action='store_false', default=True, help='False-global dist; True-local dist determined by speed')
parser.add_argument('--act-mask', action='store_true', default=False, help='False, True')
parser.add_argument('--aversion', action='store_false', default=True, help='False, True')
parser.add_argument('--aversion-coe', type=int, default=1, help='Cost coefficient')
parser.add_argument('--penalty-move', action='store_true', default=False, help='False, True')
parser.add_argument('--cost-coe', type=float, default=0, help='Cost coefficient')
parser.add_argument('--print-episodes', type=int, default=100, help='Print result every #episodes')
# update-episodes: 5, ppo-K-epochs: 5
parser.add_argument('--update-episodes', type=int, default=5, help='Update policy every #episodes')
parser.add_argument('--ppo-K-epochs', type=int, default=5, help='Update policy for K epochs in one PPO update')
parser.add_argument('--ppo-eps-clip', type=float, default=0.2, help='Clip parameter for PPO')
parser.add_argument('--mlp-hidden-dim', type=int, default=128, help='Hidden dim for MLP')
parser.add_argument('--simple-num-mlp-hidden', type=int, default=2, help='Hidden dim for MLP')
parser.add_argument('--hyper-hidden-dim', type=int, default=128, help='Hidden dim for MLP')
parser.add_argument('--z-dim', type=int, default=128, help='Hidden dim for MLP')
parser.add_argument('--scale', action='store_false', default=True, help='False-do not embedding; True-do embedding')
parser.add_argument('--bp-emb', type=str, default='dynamic', help='hyper, dynamic input')
parser.add_argument('--simple-aug', action='store_false', default=True, help='False-do not embedding; True-do embedding')
parser.add_argument('--device', type=int, default=0, help='Device')
parser.add_argument('--pol-type', type=str, default='mlp', help='MLP, MLP+ (Aug), Hyper, unified')
parser.add_argument('--alg', type=str, default='ppo', help='ppo, dqn')
parser.add_argument('--concat-meta-v-in-hidden', action='store_true', default=False, help='False, True')
parser.add_argument('--aug-concat-meta-v-in-hidden', action='store_true', default=False, help='False, True')
parser.add_argument('--init-eps', type=float, default=0.5, help='eps for e-greedy')
parser.add_argument('--final-eps', type=float, default=0.1, help='eps for e-greedy')
parser.add_argument('--eps-decay-step', type=int, default=3000, help='eps decay for e-greedy')
parser.add_argument('--reset-eps', action='store_true', default=False, help='False-do not reset; True-reset for each N')
parser.add_argument('--update-target-freq', type=int, default=200, help='target update for dqn')
parser.add_argument('--hyper-actor', action='store_true', default=True, help='False-MLP actor; True-HyperNet actor')
parser.add_argument('--hyper-critic', action='store_true', default=True, help='False-MLP critic; True-HyperNet critic')
parser.add_argument('--noise-N', action='store_true', default=False, help='False-input N; True-input N+noise')
parser.add_argument('--aug-actor', action='store_true', default=False, help='False-actor; True-aug actor')
parser.add_argument('--aug-critic', action='store_true', default=False, help='False-critic; True-aug critic')
parser.add_argument('--mask-N', action='store_true', default=False, help='False-input N; True-input 0')
parser.add_argument('--pos-emb', action='store_true', default=False, help='False-input N; True-input 0')
parser.add_argument('--pos-emb-dim', type=int, default=128, help='Hidden dim for meta variable embedding')
parser.add_argument('--max-pe-encode', type=int, default=10000, help='Hidden dim for meta variable embedding')
parser.add_argument('--meta-v-dim', type=int, default=12, help='Hidden dim for meta variable embedding')
parser.add_argument('--meta-v-emb', action='store_false', default=True, help='False-do not embedding; True-do embedding')
parser.add_argument('--lr-meta-v', type=float, default=0.0003, help='Learning rate for meta variable embedding')
parser.add_argument('--meta-v-emb-dim', type=int, default=128, help='Hidden dim for meta variable embedding')
parser.add_argument('--meta-v-grad-from-actor', action='store_true', default=False, help='False-grad from critic; True-grad from actor')
parser.add_argument('--unified-meta-v-emb', action='store_false', default=True, help='False-do not embedding; True-do embedding')
parser.add_argument('--unified-meta-v-emb-dim', type=int, default=128, help='Hidden dim for meta variable embedding')
parser.add_argument('--unified-meta-v-emb-clip', action='store_true', default=False, help='False-do not embedding; True-do embedding')
parser.add_argument('--relu-emb-in-hidden', action='store_true', default=False, help='False-do not embedding; True-do embedding')
parser.add_argument('--multi-point', action='store_false', default=True, help='False-single env; True-multiple envs')
parser.add_argument('--wait', action='store_true', default=False, help='False-single env; True-multiple envs')
parser.add_argument('--wait-time', type=int, default=7200, help='second')
parser.add_argument('--multi-point-type', type=str, default='mixenv', help='mixenv, seqenv')
parser.add_argument('--multi-point-num', type=int, default=20, help='Number of sample points')
parser.add_argument('--sample-inter', type=int, default=1, help='Number of sample points')
parser.add_argument('--mixenv-dist', type=str, default='uniform', help='uniform, poisson, seq')
parser.add_argument('--lam', type=float, default=10, help='lambda of poisson/exponential distribution')
parser.add_argument('--train-set', action='store_true', default=False, help='False-single env; True-multiple envs')
parser.add_argument('--num-train-task', type=int, default=20, help='lambda of poisson/exponential distribution')
parser.add_argument('--norm-N', action='store_true', default=False, help='False-single env; True-multiple envs')
parser.add_argument('--varied-dist', action='store_true', default=False, help='False-single env; True-multiple envs')
parser.add_argument('--reg-actor', action='store_true', default=False, help='False-actor; True-aug actor')
parser.add_argument('--reg-actor-layer', type=int, default=4, help='False-actor; True-aug actor')
parser.add_argument('--reg-critic', action='store_true', default=False, help='False-critic; True-aug critic')
parser.add_argument('--reg-critic-layer', type=int, default=4, help='False-actor; True-aug actor')
parser.add_argument('--optimizer', type=str, default='adam', help=' ')
parser.add_argument('--l1-reg', action='store_true', default=False, help=' ')
parser.add_argument('--w-clip', action='store_true', default=False, help=' ')
parser.add_argument('--w-clip-p', type=int, default=10, help=' ')
parser.add_argument('--w-decay', type=int, default=0, help=' ')

parser.add_argument('--pretrain', action='store_true', default=False, help='False-do not pre-train; True-pre-train hyper net (hyper PPO)')
parser.add_argument('--pretrain-actor', action='store_true', default=False, help='False-MLP actor; True-HyperNet actor')
parser.add_argument('--pretrain-critic', action='store_true', default=False, help='False-MLP critic; True-HyperNet critic')
parser.add_argument('--pretrain-model', action='store_true', default=False, help='False-no pretrain model; True-load pretrain model (hyper PPO)')
parser.add_argument('--pretrain-loss-type', type=str, default='mse', help='MSE, KL')
parser.add_argument('--pretrain-type', type=str, default='ol', help='ol-output level, pl-policy level')
parser.add_argument('--pretrain-batch-size', type=int, default=2048, help='Maximum steps for pretrain')
parser.add_argument('--pretrain-step', type=int, default=100000, help='Maximum steps for pretrain')
parser.add_argument('--lr-pretrain', type=float, default=0.00001, help='Learning rate for pretrain')
parser.add_argument('--distill', action='store_true', default=False, help='False-do not distill; True-do distill')
parser.add_argument('--distill-pretrain', action='store_true', default=False, help='False-do not pretrain; True-do pretrain')
parser.add_argument('--distill-target', type=str, default='mlp_plus', help='targe policy for distillation')
parser.add_argument('--distill-type', type=str, default='seq', help='seq, mix')
parser.add_argument('--distill-loss-type', type=str, default='mse', help='mse, kl')
parser.add_argument('--distill-step', type=int, default=1000000, help='Maximum steps for distillation')
parser.add_argument('--distill-batch-size', type=int, default=256, help='Batch size for distillation')
parser.add_argument('--lr-distill', type=float, default=0.00001, help='Learning rate for distillation')

# parser.add_argument('--transfer', action='store_true', default=False, help='False-do not transfer; True-transfer')
# parser.add_argument('--re-train', action='store_true', default=False, help='False-do not re-init actor; True-re-init actor')
# parser.add_argument('--fix-critic', action='store_true', default=False, help='False-critic; True-aug critic')

parser.add_argument('--transfer', action='store_true', default=False, help='False-do not transfer; True-transfer')
parser.add_argument('--fix-critic', action='store_true', default=False, help='False-critic; True-aug critic')
parser.add_argument('--re-train', action='store_true', default=False, help='False-do not re-train actor; True-re-train actor')
parser.add_argument('--re-train-episodes', type=int, default=50000, help='Episodes for retraining actor')
parser.add_argument('--fine-tunning', action='store_true', default=False, help='False-do not re-train actor; True-re-train actor')
parser.add_argument('--fine-tunning-episodes', type=int, default=1000, help='Episodes for retraining actor')
parser.add_argument('--lr-a-finetune', type=float, default=0.00001, help='Learning rate for actor')
parser.add_argument('--lr-c-finetune', type=float, default=0.0001, help='Learning rate for critic')
parser.add_argument('--transfer-from-episodes', type=int, default=2000000, help='Episodes for retraining actor')
parser.add_argument('--transfer-br-pol', action='store_true', default=False, help='False-do not transfer; True-transfer')
parser.add_argument('--transfer-agent-num', type=int, default=10, help='Number of agents in new setting')
parser.add_argument('--transfer-agent-num-min', type=int, default=10, help='Number of agents in new setting')
parser.add_argument('--transfer-agent-num-max', type=int, default=10, help='Number of agents in new setting')
parser.add_argument('--transfer-agent-num-int', type=int, default=10, help='Number of agents in new setting')
parser.add_argument('--transfer-br-max-episodes', type=int, default=100000, help='Maximum episodes for BR policy training')
parser.add_argument('--multi-point-seqenv-agent-num', type=int, default=30, help='Load trained policy with N if use seqenv type training')
parser.add_argument('--transfer-load-br-checkpoint', action='store_true', default=False, help='False-do not transfer; True-transfer')
parser.add_argument('--transfer-load-br-checkpoint-step', type=int, default=100000, help='Maximum episodes for BR policy training')


parser.add_argument('--visualize', action='store_true', default=False, help='False-do not transfer; True-transfer')
parser.add_argument('--similarity', action='store_true', default=False, help='False-do not transfer; True-transfer')
parser.add_argument('--reward-analysis', action='store_true', default=False, help='False-do not transfer; True-transfer')
parser.add_argument('--weight-analysis', action='store_true', default=False, help='False-do not transfer; True-transfer')

if __name__ == '__main__':
    device_id = [0, 1, 2, 3, 4, 5, 6, 7]
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    processname = "Taxi_train_agent{}_{}".format(args.agent_num, args.alg)
    setproctitle.setproctitle(processname)

    args.min_agent_num = args.agent_num
    args.max_agent_num = args.agent_num
    # print ('Options')
    # print ('=' * 30)
    # for k, v in vars(args).items():
    #     print(k + ': ' + str(v))
    # print ('=' * 30)

    # seeds = [1576, 1745, 1950]
    # for seed in seeds:
    #     args.seed = seed
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    agent = SLMFG(args)
    agent.train(args.init_checkpoint)
    if args.transfer:
        agent.transfer()
