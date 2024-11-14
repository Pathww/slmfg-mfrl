from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import numpy as np
from pathlib import Path
import os
import random
import pickle
import torch
from tensorboardX import SummaryWriter
from datetime import datetime
import time

from envs.taxi.env import TaxiSimulator
from envs.crowd.env import Crowd
from envs.explore.env import Explore2d

from algs.hyper_ppo import HyperPPO
from algs.ppo import PPO
from algs.dqn import DQN
from algs.aug_ppo import AugPPO
from algs.unified_hyper_ppo import UnifiedHyperPPO
from algs.simple_hyper_ppo import SimpleHyperPPO
from algs.simple_hyper_ppo_2 import SimpleHyperPPO2Layer
from algs.simple_hyper_ppo_no_head import SimpleHyperPPONoHead
from algs.simple_hyper_ppo_no_head_last_layer import SimpleHyperPPONoHeadLastLayer

from utils.utils import sample_unseen_task, get_1d_sincos_pos_embed_from_grid
from envs.utils import ids_1dto2d

class SLMFG:
    """
    SLMFG: Scaling Laws in Mean Field Games
    """
    def __init__(self, args):
        if args.min_agent_num == args.max_agent_num:
            args.agent_num = args.min_agent_num
        self.args = args
        self.render_cnt = 0
        random.seed(args.seed)
        np.random.seed(args.seed)

        if args.env_name == 'taxi':
            self.env = TaxiSimulator(args)
        elif args.env_name == 'crowd':
            self.env = Crowd(args)
        elif args.env_name == 'explore':
            self.env = Explore2d(args)
        else:
            raise ValueError(f'Unknown env {args.env_name}')

        self.N_emb = np.array([[int(x) for x in '{0:012b}'.format(i)] for i in range(4096)])
        self.N_emb_sincos = get_1d_sincos_pos_embed_from_grid(args.pos_emb_dim, np.arange(0, args.max_agent_num + 1), args.max_pe_encode)

        aug_str = ""
        hyper_str = ""
        reset_eps_str = ""
        unified_str = ""

        if args.pol_type == 'hyper':
            if args.alg == 'ppo':
                self.policy = HyperPPO(meta_v_dim=args.meta_v_dim,
                                       state_dim=self.env.obs_dim,
                                       action_dim=self.env.action_size,
                                       lr_a=args.lr_a,
                                       lr_c=args.lr_c,
                                       gamma=args.gamma,
                                       K_epochs=args.ppo_K_epochs,
                                       eps_clip=args.ppo_eps_clip,
                                       max_pe_encode=args.max_pe_encode,
                                       device=args.device,
                                       cuda=args.cuda,
                                       hidden_dim=args.mlp_hidden_dim,
                                       pos_emb=args.pos_emb,
                                       pos_emb_dim=args.pos_emb_dim,
                                       hyper_actor=args.hyper_actor,
                                       hyper_critic=args.hyper_critic,
                                       min_meta_v=args.min_agent_num,
                                       max_meta_v=args.max_agent_num,
                                       reg_actor=args.reg_actor,
                                       reg_critic=args.reg_critic,
                                       reg_actor_layer=args.reg_actor_layer,
                                       reg_critic_layer=args.reg_critic_layer,
                                       fix_critic=args.fix_critic,
                                       optimizer=args.optimizer,
                                       w_decay=args.w_decay,
                                       l1=args.l1_reg,
                                       w_clip=args.w_clip,
                                       w_clip_p=args.w_clip_p)
                hyper_str = "_a{}_c{}_noiseN{}_pret{}_a{}_c{}_{}_{}"\
                    .format(int(args.hyper_actor), int(args.hyper_critic), int(args.noise_N), int(args.pretrain), int(args.pretrain_actor),
                            int(args.pretrain_critic), args.pretrain_loss_type, args.pretrain_step)
        elif args.pol_type == 'mlp':
            if args.alg == 'ppo':
                self.policy = PPO(state_dim=self.env.obs_dim,
                                  action_dim=self.env.action_size,
                                  hidden_dim=args.mlp_hidden_dim,
                                  lr_a=args.lr_a,
                                  lr_c=args.lr_c,
                                  gamma=args.gamma,
                                  K_epochs=args.ppo_K_epochs,
                                  eps_clip=args.ppo_eps_clip,
                                  device=args.device,
                                  cuda=args.cuda, 
                                  use_mf=args.use_mf)
            if args.alg == 'dqn':
                self.policy = DQN(state_dim=self.env.obs_dim,
                                  action_dim=self.env.action_size,
                                  hidden_dim=args.mlp_hidden_dim,
                                  lr=args.lr,
                                  gamma=args.gamma,
                                  init_eps = args.init_eps,
                                  final_eps = args.final_eps,
                                  eps_decay_step = args.eps_decay_step,
                                  device=args.device,
                                  cuda=args.cuda, 
                                  use_mf=args.use_mf)
        elif args.pol_type == 'mlp_plus':
            if args.alg == 'ppo':
                self.policy = AugPPO(meta_v_dim=args.meta_v_dim,
                                     meta_v_emb=args.meta_v_emb,
                                     meta_v_emb_dim=args.meta_v_emb_dim,
                                     state_dim=self.env.obs_dim,
                                     action_dim=self.env.action_size,
                                     hidden_dim=args.mlp_hidden_dim,
                                     pos_emb=args.pos_emb,
                                     pos_emb_dim=args.pos_emb_dim,
                                     lr_a=args.lr_a,
                                     lr_c=args.lr_c,
                                     gamma=args.gamma,
                                     K_epochs=args.ppo_K_epochs,
                                     eps_clip=args.ppo_eps_clip,
                                     device=args.device,
                                     cuda=args.cuda,
                                     min_meta_v=args.min_agent_num,
                                     max_meta_v=args.max_agent_num,
                                     concat_meta_v_in_hidden=args.aug_concat_meta_v_in_hidden)
                aug_str = "_a{}_c{}_noiseN{}_emb{}_dim{}{}{}"\
                    .format(int(args.aug_actor), int(args.aug_critic), int(args.noise_N), int(args.meta_v_emb),
                            args.meta_v_emb_dim, "_mask_N1" if args.mask_N else "", "_ch1" if args.aug_concat_meta_v_in_hidden else "")
        elif args.pol_type == 'unified':
            if args.alg == 'ppo':
                self.policy = UnifiedHyperPPO(meta_v_dim=args.meta_v_dim,
                                              meta_v_emb=args.unified_meta_v_emb,
                                              meta_v_emb_dim=args.unified_meta_v_emb_dim,
                                              state_dim=self.env.obs_dim,
                                              action_dim=self.env.action_size,
                                              lr_a=args.lr_a,
                                              lr_c=args.lr_c,
                                              gamma=args.gamma,
                                              K_epochs=args.ppo_K_epochs,
                                              eps_clip=args.ppo_eps_clip,
                                              max_pe_encode=args.max_pe_encode,
                                              device=args.device,
                                              cuda=args.cuda,
                                              hidden_dim=args.mlp_hidden_dim,
                                              pos_emb=args.pos_emb,
                                              pos_emb_dim=args.pos_emb_dim,
                                              min_meta_v=args.min_agent_num,
                                              max_meta_v=args.max_agent_num,
                                              optimizer=args.optimizer,
                                              w_decay=args.w_decay,
                                              concat_meta_v_in_hidden=args.concat_meta_v_in_hidden,
                                              clip_emb=args.unified_meta_v_emb_clip)
                meta_v_emb_str = ""
                if args.unified_meta_v_emb:
                    meta_v_emb_str = "_emb1_dim{}{}".format(args.unified_meta_v_emb_dim, "_ce1" if args.unified_meta_v_emb_clip else "")
                unified_str = "_a{}_c{}_noiseN{}{}_pret{}_a{}_c{}_{}_{}{}"\
                    .format(int(args.hyper_actor), int(args.hyper_critic), int(args.noise_N), meta_v_emb_str, int(args.pretrain),
                            int(args.pretrain_actor), int(args.pretrain_critic), args.pretrain_loss_type, args.pretrain_step,
                            "_pl" if args.pretrain_type == 'pl' else "")
        elif args.pol_type == 'simple':
            if args.alg == 'ppo':
                self.policy = SimpleHyperPPO(meta_v_dim=args.meta_v_dim,
                                             meta_v_emb=args.unified_meta_v_emb,
                                             meta_v_emb_dim=args.unified_meta_v_emb_dim,
                                             state_dim=self.env.obs_dim,
                                             action_dim=self.env.action_size,
                                             lr_a=args.lr_a,
                                             lr_c=args.lr_c,
                                             gamma=args.gamma,
                                             K_epochs=args.ppo_K_epochs,
                                             eps_clip=args.ppo_eps_clip,
                                             max_pe_encode=args.max_pe_encode,
                                             device=args.device,
                                             cuda=args.cuda,
                                             hidden_dim=args.mlp_hidden_dim,
                                             pos_emb=args.pos_emb,
                                             pos_emb_dim=args.pos_emb_dim,
                                             hyper_actor=args.hyper_actor,
                                             hyper_critic=args.hyper_critic,
                                             min_meta_v=args.min_agent_num,
                                             max_meta_v=args.max_agent_num,
                                             reg_actor=args.reg_actor,
                                             reg_critic=args.reg_critic,
                                             reg_actor_layer=args.reg_actor_layer,
                                             reg_critic_layer=args.reg_critic_layer,
                                             fix_critic=args.fix_critic,
                                             optimizer=args.optimizer,
                                             w_decay=args.w_decay,
                                             l1=args.l1_reg,
                                             w_clip=args.w_clip,
                                             w_clip_p=args.w_clip_p,
                                             concat_meta_v_in_hidden=args.concat_meta_v_in_hidden,
                                             clip_emb=args.unified_meta_v_emb_clip)
                meta_v_emb_str = ""
                if args.unified_meta_v_emb:
                    meta_v_emb_str = "_emb1_dim{}{}".format(args.unified_meta_v_emb_dim, "_ce1" if args.unified_meta_v_emb_clip else "")
                ppo_train_config = ""
                if args.update_episodes != 5:
                    ppo_train_config = "_te{}_{}".format(args.update_episodes, args.ppo_K_epochs)
                unified_str = "_a{}_c{}_noiseN{}{}_pret{}_a{}_c{}_{}_{}{}"\
                    .format(int(args.hyper_actor), int(args.hyper_critic), int(args.noise_N), meta_v_emb_str, int(args.pretrain),
                            int(args.pretrain_actor), int(args.pretrain_critic), args.pretrain_loss_type, args.pretrain_step,
                            ppo_train_config)
        elif args.pol_type == 'simple2':
            if args.alg == 'ppo':
                self.policy = SimpleHyperPPO2Layer(meta_v_dim=args.meta_v_dim,
                                                   meta_v_emb=args.unified_meta_v_emb,
                                                   meta_v_emb_dim=args.unified_meta_v_emb_dim,
                                                   state_dim=self.env.obs_dim,
                                                   action_dim=self.env.action_size,
                                                   lr_a=args.lr_a,
                                                   lr_c=args.lr_c,
                                                   lr_a_finetune=args.lr_a_finetune,
                                                   lr_c_finetune=args.lr_c_finetune,
                                                   gamma=args.gamma,
                                                   K_epochs=args.ppo_K_epochs,
                                                   eps_clip=args.ppo_eps_clip,
                                                   max_pe_encode=args.max_pe_encode,
                                                   device=args.device,
                                                   cuda=args.cuda,
                                                   hyper_hidden_dim=args.hyper_hidden_dim,
                                                   z_dim=args.z_dim,
                                                   dynamic_hidden_dim=args.mlp_hidden_dim,
                                                   pos_emb=args.pos_emb,
                                                   pos_emb_dim=args.pos_emb_dim,
                                                   hyper_actor=args.hyper_actor,
                                                   hyper_critic=args.hyper_critic,
                                                   min_meta_v=args.min_agent_num,
                                                   max_meta_v=args.max_agent_num,
                                                   reg_actor=args.reg_actor,
                                                   reg_critic=args.reg_critic,
                                                   reg_actor_layer=args.reg_actor_layer,
                                                   reg_critic_layer=args.reg_critic_layer,
                                                   fix_critic=args.fix_critic,
                                                   optimizer=args.optimizer,
                                                   w_decay=args.w_decay,
                                                   l1=args.l1_reg,
                                                   w_clip=args.w_clip,
                                                   w_clip_p=args.w_clip_p,
                                                   concat_meta_v_in_hidden=args.concat_meta_v_in_hidden,
                                                   clip_emb=args.unified_meta_v_emb_clip,
                                                   num_hidden=args.simple_num_mlp_hidden,
                                                   scale=args.scale,
                                                   bp_emb=args.bp_emb,
                                                   aug=args.simple_aug,
                                                   relu_emb_in_hidden=args.relu_emb_in_hidden)
                meta_v_emb_str = ""
                bp_emb_str = ""
                if args.bp_emb == 'hyper':
                    bp_emb_str = "_bpe_h"
                elif args.bp_emb == 'all':
                    bp_emb_str = "_bpe_a"
                if args.unified_meta_v_emb:
                    meta_v_emb_str = "_emb1_dim{}{}{}".format(args.unified_meta_v_emb_dim, "_ce1" if args.unified_meta_v_emb_clip else "", bp_emb_str)
                ppo_train_config = ""
                if args.update_episodes != 5:
                    ppo_train_config = "_te{}_{}".format(args.update_episodes, args.ppo_K_epochs)
                z_dim_str = ""
                if args.z_dim != 128:
                    z_dim_str = "_z{}".format(args.z_dim)
                unified_str = "_a{}_c{}_noiseN{}{}_pret{}_a{}_c{}_{}_{}_ln{}{}{}{}"\
                    .format(int(args.hyper_actor), int(args.hyper_critic), int(args.noise_N), meta_v_emb_str, int(args.pretrain),
                            int(args.pretrain_actor), int(args.pretrain_critic), args.pretrain_loss_type, args.pretrain_step,
                            args.simple_num_mlp_hidden, ppo_train_config, "" if args.simple_aug else "_aug0", z_dim_str)
        elif args.pol_type == 'simplenohead':
            if args.alg == 'ppo':
                self.policy = SimpleHyperPPONoHead(meta_v_dim=args.meta_v_dim,
                                                   meta_v_emb=args.unified_meta_v_emb,
                                                   meta_v_emb_dim=args.unified_meta_v_emb_dim,
                                                   state_dim=self.env.obs_dim,
                                                   action_dim=self.env.action_size,
                                                   lr_a=args.lr_a,
                                                   lr_c=args.lr_c,
                                                   gamma=args.gamma,
                                                   K_epochs=args.ppo_K_epochs,
                                                   eps_clip=args.ppo_eps_clip,
                                                   max_pe_encode=args.max_pe_encode,
                                                   device=args.device,
                                                   cuda=args.cuda,
                                                   hyper_hidden_dim=args.hyper_hidden_dim,
                                                   dynamic_hidden_dim=args.mlp_hidden_dim,
                                                   pos_emb=args.pos_emb,
                                                   pos_emb_dim=args.pos_emb_dim,
                                                   hyper_actor=args.hyper_actor,
                                                   hyper_critic=args.hyper_critic,
                                                   min_meta_v=args.min_agent_num,
                                                   max_meta_v=args.max_agent_num,
                                                   reg_actor=args.reg_actor,
                                                   reg_critic=args.reg_critic,
                                                   reg_actor_layer=args.reg_actor_layer,
                                                   reg_critic_layer=args.reg_critic_layer,
                                                   fix_critic=args.fix_critic,
                                                   optimizer=args.optimizer,
                                                   w_decay=args.w_decay,
                                                   l1=args.l1_reg,
                                                   w_clip=args.w_clip,
                                                   w_clip_p=args.w_clip_p,
                                                   concat_meta_v_in_hidden=args.concat_meta_v_in_hidden,
                                                   clip_emb=args.unified_meta_v_emb_clip,
                                                   num_hidden=args.simple_num_mlp_hidden,
                                                   scale=args.scale,
                                                   aug=args.simple_aug)
                meta_v_emb_str = ""
                if args.unified_meta_v_emb:
                    meta_v_emb_str = "_emb1_dim{}{}".format(args.unified_meta_v_emb_dim, "_ce1" if args.unified_meta_v_emb_clip else "")
                ppo_train_config = ""
                if args.update_episodes != 5:
                    ppo_train_config = "_te{}_{}".format(args.update_episodes, args.ppo_K_epochs)
                unified_str = "_a{}_c{}_noiseN{}{}_pret{}_a{}_c{}_{}_{}_ln{}{}{}"\
                    .format(int(args.hyper_actor), int(args.hyper_critic), int(args.noise_N), meta_v_emb_str, int(args.pretrain),
                            int(args.pretrain_actor), int(args.pretrain_critic), args.pretrain_loss_type, args.pretrain_step,
                            args.simple_num_mlp_hidden, ppo_train_config, "" if args.simple_aug else "_aug0")
        elif args.pol_type == 'simplenhll':
            if args.alg == 'ppo':
                self.policy = SimpleHyperPPONoHeadLastLayer(meta_v_dim=args.meta_v_dim,
                                                            meta_v_emb=args.unified_meta_v_emb,
                                                            meta_v_emb_dim=args.unified_meta_v_emb_dim,
                                                            state_dim=self.env.obs_dim,
                                                            action_dim=self.env.action_size,
                                                            lr_a=args.lr_a,
                                                            lr_c=args.lr_c,
                                                            gamma=args.gamma,
                                                            K_epochs=args.ppo_K_epochs,
                                                            eps_clip=args.ppo_eps_clip,
                                                            max_pe_encode=args.max_pe_encode,
                                                            device=args.device,
                                                            cuda=args.cuda,
                                                            hyper_hidden_dim=args.hyper_hidden_dim,
                                                            dynamic_hidden_dim=args.mlp_hidden_dim,
                                                            pos_emb=args.pos_emb,
                                                            pos_emb_dim=args.pos_emb_dim,
                                                            hyper_actor=args.hyper_actor,
                                                            hyper_critic=args.hyper_critic,
                                                            min_meta_v=args.min_agent_num,
                                                            max_meta_v=args.max_agent_num,
                                                            reg_actor=args.reg_actor,
                                                            reg_critic=args.reg_critic,
                                                            reg_actor_layer=args.reg_actor_layer,
                                                            reg_critic_layer=args.reg_critic_layer,
                                                            fix_critic=args.fix_critic,
                                                            optimizer=args.optimizer,
                                                            w_decay=args.w_decay,
                                                            l1=args.l1_reg,
                                                            w_clip=args.w_clip,
                                                            w_clip_p=args.w_clip_p,
                                                            concat_meta_v_in_hidden=args.concat_meta_v_in_hidden,
                                                            clip_emb=args.unified_meta_v_emb_clip,
                                                            num_hidden=args.simple_num_mlp_hidden,
                                                            scale=args.scale,
                                                            aug=args.simple_aug)
                meta_v_emb_str = ""
                if args.unified_meta_v_emb:
                    meta_v_emb_str = "_emb1_dim{}{}".format(args.unified_meta_v_emb_dim, "_ce1" if args.unified_meta_v_emb_clip else "")
                ppo_train_config = ""
                if args.update_episodes != 5:
                    ppo_train_config = "_te{}_{}".format(args.update_episodes, args.ppo_K_epochs)
                unified_str = "_a{}_c{}_noiseN{}{}_pret{}_a{}_c{}_{}_{}_ln{}{}{}"\
                    .format(int(args.hyper_actor), int(args.hyper_critic), int(args.noise_N), meta_v_emb_str, int(args.pretrain),
                            int(args.pretrain_actor), int(args.pretrain_critic), args.pretrain_loss_type, args.pretrain_step,
                            args.simple_num_mlp_hidden, ppo_train_config, "" if args.simple_aug else "_aug0")

        if args.transfer or args.visualize:
            # BR policy used to evaluate the performance of transfer
            if args.alg == 'ppo':
                self.br_policy = PPO(state_dim=self.env.obs_dim,
                                     action_dim=self.env.action_size,
                                     hidden_dim=args.mlp_hidden_dim,
                                     lr_a=args.lr_a,
                                     lr_c=args.lr_c,
                                     gamma=args.gamma,
                                     K_epochs=args.ppo_K_epochs,
                                     eps_clip=args.ppo_eps_clip,
                                     device=args.device,
                                     cuda=args.cuda)
            if args.alg == 'dqn':
                self.policy = DQN(state_dim=self.env.obs_dim,
                                  action_dim=self.env.action_size,
                                  hidden_dim=args.mlp_hidden_dim,
                                  lr=args.lr,
                                  gamma=args.gamma,
                                  init_eps = args.init_eps,
                                  final_eps = args.final_eps,
                                  eps_decay_step = args.eps_decay_step,
                                  device=args.device,
                                  cuda=args.cuda)
        self.multi_point_str = ""
        agent_num_str = "_an{}_to_{}".format(args.min_agent_num, args.max_agent_num)
        mixenv_dist_para = ""
        if args.mixenv_dist == 'poisson' or args.mixenv_dist == 'exponent':
            mixenv_dist_para = "_lam{}".format(args.lam)
        self.multi_point_str = "_mp1_mixenv_{}{}{}".format(args.mixenv_dist, mixenv_dist_para, "" if args.sample_inter == 1 else "_si{}".format(args.sample_inter))
        self.train_set_str = ""
        if args.train_set:
            self.train_set_str = "_ts{}_num{}".format(int(args.train_set), args.num_train_task)
        self.norm_N_str = ""
        if args.norm_N:
            self.norm_N_str = "_normN{}".format(int(args.norm_N))
        self.meta_v_str = ""
        if args.meta_v_dim > 1:
            self.meta_v_str = "_meta_dim{}".format(args.meta_v_dim)
        self.pos_emb_str = ""
        if args.pos_emb:
            assert (args.pol_type == 'unified' or args.pol_type == 'simple' or args.pol_type == 'simple2'
                    or args.pol_type == 'simplenohead' or args.pol_type == 'simplenhll') and args.unified_meta_v_emb, \
                "Positional encoding is only supported for unified/simple PPO and must be used with meta_v_emb."
            self.pos_emb_str = "_pe1_me{}{}".format(args.max_pe_encode, "_scale0" if not args.scale else "")
        self.new_reg_str = ""
        if args.reg_actor or args.reg_critic:
            self.new_reg_str = "_reg_a{}_l{}_c{}_l{}".format(int(args.reg_actor), args.reg_actor_layer, int(args.reg_critic), args.reg_critic_layer)
        self.old_reg_str = ""
        if args.l1_reg:
            self.old_reg_str += "_l1{}".format(int(args.l1_reg))
        if args.w_clip:
            self.old_reg_str += "_wc{}_{}".format(int(args.w_clip), args.w_clip_p)
        self.map_type_str = ""
        if args.env_name == 'taxi' or args.env_name == 'taxi_toy':
            self.map_type_str += "_on{}_{}_rm_{}".format(args.order_num, args.order_dist, args.reward_method)
            if args.map_str == "grid":
                self.map_type_str += "_map{}x{}".format(args.map_M, args.map_N)
            else:
                self.map_type_str += "_{}".format(args.map_str)
        self.aversion_str = ""
        if args.aversion_coe != 1:
            self.aversion_str = "_c{}".format(args.aversion_coe)
        c_h_str = ""
        if (args.pol_type == 'unified' or args.pol_type == 'simple' or args.pol_type == 'simple2'
            or args.pol_type == 'simplenohead' or args.pol_type == 'simplenhll') and args.concat_meta_v_in_hidden:
            c_h_str = "_ch1"
            if args.unified_meta_v_emb and args.relu_emb_in_hidden:
                c_h_str += "_rh1"
        self.run_name = "seed{}{}{}_av{}{}_pol_{}{}{}{}{}_alg_{}{}_step{}_lobs{}{}_h{}{}{}{}{}{}{}"\
            .format(args.seed, agent_num_str, self.map_type_str, int(args.aversion), self.aversion_str, args.pol_type, hyper_str,
                    aug_str, unified_str, c_h_str, args.alg, reset_eps_str, args.max_episodes, int(args.local_obs), self.meta_v_str,
                    args.mlp_hidden_dim, self.multi_point_str, self.train_set_str, self.norm_N_str, self.old_reg_str, self.new_reg_str,
                    self.pos_emb_str)
        if not args.visualize and not args.transfer and not args.similarity \
                and not args.reward_analysis and not args.weight_analysis:
            self.writer = SummaryWriter(comment=self.run_name + '_' + str(self.args.env_name))
        else:
            self.writer = None

        self.record_dir = "".join([args.record, "/", args.env_name, "/", self.run_name])
        p = Path(self.record_dir)
        if not p.is_dir():
            p.mkdir(parents=True)
        self.checkpoint_dir = "".join([self.record_dir, '/checkpoints/'])
        p = Path(self.checkpoint_dir)
        if not p.is_dir():
            p.mkdir(parents=True)

        if self.args.render:
            self.render_dir = "".join([self.record_dir, '/render/'])
            p = Path(self.render_dir)
            if not p.is_dir():
                p.mkdir(parents=True)
                
        self.pret_dir = "".join([args.record, "/", args.env_name, "/pret_pol/"])
        p = Path(self.pret_dir)
        if not p.is_dir():
            p.mkdir(parents=True)

        if args.wait:
            print('Start:  ', datetime.now().replace(microsecond=0))
            print('Wait(s):', args.wait_time)
            print(self.run_name)
            time.sleep(args.wait_time)

    def rollout(self, meta_v=None, env=None, ret_prob=False, pos_emb=None, finetune=False, store_tuple=True):
        episode_reward = 0
        episode_dist = []
        action_probs = []
        if env is None:
            cur_env = self.env
        else:
            cur_env = env
        cur_env.reset()
        episode_dist.append(cur_env.get_agent_dist())
        obs, act_masks = cur_env.get_obs(0)
        dones = [False] * cur_env.agent_num

        if self.args.render:
            render_record = np.zeros((self.args.episode_len, cur_env.agent_num, 2))
            self.render_cnt += 1

        if self.args.use_mf:
            agent_num = cur_env.agent_num
            action_num = 5
            former_act_prob = np.zeros((agent_num, action_num))

        for t in range(self.args.episode_len):
            if meta_v is not None:
                if ret_prob:
                    actions, probs = self.policy.select_action(meta_v, obs, ret_prob=ret_prob, pos_emb=pos_emb)
                    action_probs.append(probs)
                else:
                    if finetune:
                        actions = self.policy.select_action_finetune(meta_v, obs, pos_emb=pos_emb)
                    else:
                        actions = self.policy.select_action(meta_v, obs, pos_emb=pos_emb)
            else:
                if self.args.use_mf:
                    actions = self.policy.select_action(obs, former_act_prob=former_act_prob, store_tuple=store_tuple)
                else:
                    actions = self.policy.select_action(obs, store_tuple=store_tuple)
            
            if self.args.use_mf:
                # 全部智能体
                former_act_prob = np.mean(list(map(lambda x: np.eye(action_num)[x], actions)), axis=0)
                former_act_prob = np.tile(former_act_prob, (agent_num, 1))
                # 除决策智能体本身以外的智能体
                # former_act_sum = np.sum(list(map(lambda x: np.eye(action_num)[x], actions)), axis=0)
                # former_act_prob = np.empty((0, action_num))
                # for i in range(agent_num):
                #     tmp = former_act_sum - np.eye(action_num)[actions[i]]
                #     former_act_prob = np.vstack((former_act_prob, tmp))
                # former_act_prob /= agent_num-1

            rewards, agent_final_node_id = cur_env.step(t + 1, actions, act_masks)

            if self.args.render:
                i, j = ids_1dto2d(agent_final_node_id, self.args.map_M, self.args.map_N)
                i = np.reshape(i, (cur_env.agent_num,1))
                j = np.reshape(j, (cur_env.agent_num,1))
                render_record[t] = np.concatenate((i,j),axis=1)
                render_orders = cur_env.get_render_orders()

            obs, act_masks = cur_env.get_obs(t + 1)
            if t + 1 == self.args.episode_len:
                dones = [True] * cur_env.agent_num

            if self.args.alg == 'ppo' and store_tuple:
                self.policy.buffer.rewards.append(rewards[0])
                self.policy.buffer.is_terminals.append(dones[0])
            if self.args.alg == 'dqn' and store_tuple:
                if self.args.use_mf:
                    state_next = np.concatenate((obs, former_act_prob), axis=1)
                else:
                    state_next = obs
                self.policy.buffer.append_state_next(state_next[0])
                self.policy.buffer.append_reward(rewards[0])
                self.policy.buffer.append_is_terminal(dones[0])
                self.policy.buffer.add_cnt()
            
            episode_reward += rewards[0]
            episode_dist.append(cur_env.get_agent_dist())

        if ret_prob:
            return episode_reward, episode_dist, action_probs

        if self.args.render and self.render_cnt % self.args.render_every == 0:
            np.savez('{}rollout_{}'.format(self.render_dir, self.render_cnt), agents=render_record, orders=render_orders)
        return episode_reward, episode_dist

    def train(self, init_checkpoint=0):
        args = self.args
        cuda = args.cuda
        if init_checkpoint:
            if args.pretrain and args.pretrain_model:
                self.policy.load(self.checkpoint_dir + 'policy_{}_start_from_seed1576_pret.pth'.format(init_checkpoint))
            else:
                print('Load Hyper/Unified PPO checkpoint {}: {}'.format(init_checkpoint, self.checkpoint_dir + 'policy_' + str(init_checkpoint) + '.pth'))
                self.policy.load(self.checkpoint_dir + 'policy_' + str(init_checkpoint) + '.pth')
            start_episode = init_checkpoint
        else:
            start_episode = 0

        if init_checkpoint == 0 and args.pretrain:
            if args.pretrain_type == 'pl':
                ppo = AugPPO(meta_v_dim=args.meta_v_dim,
                             meta_v_emb=args.unified_meta_v_emb,
                             meta_v_emb_dim=args.unified_meta_v_emb_dim,
                             state_dim=self.env.obs_dim,
                             action_dim=self.env.action_size,
                             hidden_dim=args.mlp_hidden_dim,
                             pos_emb=args.pos_emb,
                             pos_emb_dim=args.pos_emb_dim,
                             lr_a=args.lr_a,
                             lr_c=args.lr_c,
                             gamma=args.gamma,
                             K_epochs=args.ppo_K_epochs,
                             eps_clip=args.ppo_eps_clip,
                             device=args.device,
                             cuda=args.cuda,
                             min_meta_v=args.min_agent_num,
                             max_meta_v=args.max_agent_num,
                             concat_meta_v_in_hidden=args.concat_meta_v_in_hidden)
            else:
                ppo = PPO(state_dim=self.env.obs_dim,
                          action_dim=self.env.action_size,
                          hidden_dim=args.mlp_hidden_dim,
                          lr_a=args.lr_a,
                          lr_c=args.lr_c,
                          gamma=args.gamma,
                          K_epochs=args.ppo_K_epochs,
                          eps_clip=args.ppo_eps_clip,
                          device=args.device,
                          cuda=cuda)
            if args.pol_type == 'hyper':
                pret_pol = "".join([self.pret_dir, 'pretrain_policy.pth'])
            elif args.pol_type == 'unified' or args.pol_type == 'simple' or args.pol_type == 'simple2' \
                    or args.pol_type == 'simplenohead' or args.pol_type == 'simplenhll':
                if not args.unified_meta_v_emb:
                    if args.concat_meta_v_in_hidden:
                        if args.pretrain_type == 'pl':
                            pret_pol = "".join([self.pret_dir, 'pretrain_{}_pl_policy.pth'.format(args.pol_type)])
                        else:
                            pret_pol = "".join([self.pret_dir, 'pretrain_{}_policy.pth'.format(args.pol_type)])
                    else:
                        if args.pretrain_type == 'pl':
                            pret_pol = "".join([self.pret_dir, 'pretrain_{}_no_ch_pl_policy.pth'.format(args.pol_type)])
                        else:
                            pret_pol = "".join([self.pret_dir, 'pretrain_{}_no_ch_policy.pth'.format(args.pol_type)])
                else:
                    if not args.pos_emb:
                        if args.concat_meta_v_in_hidden:
                            if args.pretrain_type == 'pl':
                                pret_pol = "".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pl_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)])
                            else:
                                pret_pol = "".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)])
                        else:
                            if args.pretrain_type == 'pl':
                                pret_pol = "".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_no_ch_pl_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)])
                            else:
                                pret_pol = "".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_no_ch_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)])
                    else:
                        if args.concat_meta_v_in_hidden:
                            if args.pretrain_type == 'pl':
                                pret_pol = "".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pos_emb1_pl_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)])
                            else:
                                pret_pol = "".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pos_emb1_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)])
                        else:
                            if args.pretrain_type == 'pl':
                                pret_pol = "".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pos_emb1_no_ch_pl_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)])
                            else:
                                pret_pol = "".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pos_emb1_no_ch_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)])
            if os.path.exists(pret_pol):
                print('Load Pre-Trained Hyper PPO: ', pret_pol)
                self.policy.load(pret_pol)
            else:
                self.policy.pretrain(args.min_agent_num, args.max_agent_num, ppo, args.pretrain_batch_size,
                                     args.pretrain_step, args.lr_pretrain, self.writer, args.seed, self.env,
                                     args.episode_len, args.norm_N, args.pretrain_loss_type, self.multi_point_str, args.pretrain_type)
                if args.pol_type == 'hyper':
                    self.policy.save("".join([self.pret_dir, 'pretrain_policy.pth']))
                elif args.pol_type == 'unified' or args.pol_type == 'simple' or args.pol_type == 'simple2' \
                        or args.pol_type == 'simplenohead' or args.pol_type == 'simplenhll':
                    if not args.unified_meta_v_emb:
                        if args.concat_meta_v_in_hidden:
                            if args.pretrain_type == 'pl':
                                self.policy.save("".join([self.pret_dir, 'pretrain_{}_pl_policy.pth'.format(args.pol_type)]))
                            else:
                                self.policy.save("".join([self.pret_dir, 'pretrain_{}_policy.pth'.format(args.pol_type)]))
                        else:
                            if args.pretrain_type == 'pl':
                                self.policy.save("".join([self.pret_dir, 'pretrain_{}_no_ch_pl_policy.pth'.format(args.pol_type)]))
                            else:
                                self.policy.save("".join([self.pret_dir, 'pretrain_{}_no_ch_policy.pth'.format(args.pol_type)]))
                    else:
                        if not args.pos_emb:
                            if args.concat_meta_v_in_hidden:
                                if args.pretrain_type == 'pl':
                                    self.policy.save("".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pl_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)]))
                                else:
                                    self.policy.save("".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)]))
                            else:
                                if args.pretrain_type == 'pl':
                                    self.policy.save("".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_no_ch_pl_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)]))
                                else:
                                    self.policy.save("".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_no_ch_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)]))
                        else:
                            if args.concat_meta_v_in_hidden:
                                if args.pretrain_type == 'pl':
                                    self.policy.save("".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pos_emb1_pl_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)]))
                                else:
                                    self.policy.save("".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pos_emb1_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)]))
                            else:
                                if args.pretrain_type == 'pl':
                                    self.policy.save("".join([self.pret_dir, 'pretrain_{}_with_embedding_{}_pos_emb1_no_ch_pl_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)]))
                                else:
                                    self.policy.save("".join([self.pret_dir,'pretrain_{}_with_embedding_{}_pos_emb1_no_ch_policy.pth'.format(args.pol_type, args.unified_meta_v_emb_dim)]))

        start_time = datetime.now().replace(microsecond=0)

        episode_cnt = start_episode
        if episode_cnt == 0:
            cumulative_rewards = []
            sample_point_cnt = np.zeros(args.max_agent_num + 1)
            if args.train_set:
                train_set = sample_unseen_task(args.min_agent_num, args.max_agent_num + 1, num_of_unseen=args.num_train_task, method=args.mixenv_dist, lam=args.lam)
            else:
                train_set = []
        else:
            if args.pretrain and args.pretrain_model:
                data = pickle.load(open(self.record_dir + '/' + 'cumulative_rewards_start_from_seed1576_pret.pik', 'rb'))
            else:
                data = pickle.load(open(self.record_dir + '/' + 'cumulative_rewards.pik', 'rb'))
            cumulative_rewards = data['cumulative_rewards']
            sample_point_cnt = data['sample_point_cnt']
            train_set = data['train_set']
        # if args.min_agent_num == args.max_agent_num:
        #     a_num = args.min_agent_num
        # else:
        #     a_num = args.min_agent_num - 1
        while episode_cnt < args.max_episodes:
            start = datetime.now().timestamp()
            if args.min_agent_num != args.max_agent_num:
                # sample a new env
                if episode_cnt % args.sample_inter == 0:
                    if args.train_set:
                        np.random.seed(episode_cnt)
                        a_num = np.random.choice(train_set)
                    else:
                        if args.mixenv_dist == 'uniform':
                            np.random.seed(episode_cnt)
                            a_num = np.random.randint(args.min_agent_num, args.max_agent_num + 1)
                        elif args.mixenv_dist == 'poisson':
                            np.random.seed(episode_cnt)
                            while True:
                                a_num = np.random.poisson(args.lam, size=1)
                                if args.min_agent_num <= a_num <= args.max_agent_num:
                                    break
                        elif args.mixenv_dist == 'exponent':
                            np.random.seed(episode_cnt)
                            while True:
                                a_num = int(np.round(np.random.exponential(args.lam, size=1)))
                                if args.min_agent_num <= a_num <= args.max_agent_num:
                                    break
                        elif args.mixenv_dist == 'seq':
                            if a_num < args.max_agent_num:
                                a_num += 1
                            else:
                                a_num = args.min_agent_num
                        else:
                            raise ValueError(f"Unknown env sample method: {args.mexenv_dist}")
                sample_point_cnt[int(a_num)] += 1
                self.env.reset_agent_pos(a_num)
            cur_agent_num = self.env.agent_num
            # print('a num: ', cur_agent_num)
            # input('dddddfffff')
            if args.pol_type == 'hyper' or args.pol_type == 'mlp_plus' or args.pol_type == 'unified' \
                    or args.pol_type == 'simple' or args.pol_type == 'simple2' or args.pol_type == 'simplenohead'\
                    or args.pol_type == 'simplenhll':
                if args.mask_N:
                    meta_v = torch.zeros(cur_agent_num, dtype=torch.float32).unsqueeze(1)
                else:
                    if args.noise_N:
                        meta_v = torch.FloatTensor(cur_agent_num - 0.5 + np.random.rand(cur_agent_num)).unsqueeze(1)
                    else:
                        if args.meta_v_dim > 1:
                            meta_v = torch.FloatTensor(np.array([self.N_emb[cur_agent_num]] * cur_agent_num))
                        else:
                            meta_v = torch.FloatTensor(np.array([cur_agent_num] * cur_agent_num)).unsqueeze(1)
                if args.norm_N:
                    meta_v = (meta_v - args.min_agent_num) / (args.max_agent_num - args.min_agent_num)
                if cuda:
                    meta_v = meta_v.to(args.device)
            else:
                meta_v = None
            if args.pos_emb:
                pos_emb = torch.FloatTensor(np.array([self.N_emb_sincos[cur_agent_num]] * cur_agent_num))
                if cuda:
                    pos_emb = pos_emb.to(args.device)
            else:
                pos_emb = None
            reward, _ = self.rollout(meta_v, pos_emb=pos_emb)
            cumulative_rewards.append(reward)
            self.writer.add_scalar('reward', reward, episode_cnt)
            if args.alg == 'ppo' or args.alg == 'dqn':
                if (episode_cnt + 1) % self.args.update_episodes == 0:
                    self.policy.update()
            end = datetime.now().timestamp()
            episode_time = (end - start)  # second
            if episode_cnt % args.print_episodes == 0 or episode_cnt == args.max_episodes - 1:
                print("(Train) Seed:{}, env:{}{}, Epi:#{}/{}, AvgR:{:.4f}, Pol:{}-{}, N:{}, MP:{}, T:{:.3f}"
                      .format(args.seed, args.env_name, self.map_type_str, episode_cnt, args.max_episodes, reward,
                              args.pol_type, args.alg, cur_agent_num, self.multi_point_str, episode_time))
            if (episode_cnt + 1) % args.checkpoint_episodes == 0 or (episode_cnt + 1) == args.max_episodes:
                if args.pretrain and args.pretrain_model:
                    self.policy.save("".join([self.checkpoint_dir, 'policy_', str(episode_cnt + 1), '_start_from_seed1576_pret.pth']))
                else:
                    self.policy.save("".join([self.checkpoint_dir, 'policy_', str(episode_cnt + 1), '.pth']))
            if (episode_cnt + 1) % 2000 == 0:
                if args.pretrain and args.pretrain_model:
                    pickle.dump({'cumulative_rewards': cumulative_rewards,
                                 'sample_point_cnt': sample_point_cnt,
                                 'train_set': train_set},
                                open(self.record_dir + '/cumulative_rewards_start_from_seed1576_pret.pik', 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)
                else:
                    pickle.dump({'cumulative_rewards': cumulative_rewards,
                                 'sample_point_cnt': sample_point_cnt,
                                 'train_set': train_set},
                                open(self.record_dir + '/cumulative_rewards.pik', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            episode_cnt += 1

        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        if args.pretrain and args.pretrain_model:
            pickle.dump({'cumulative_rewards': cumulative_rewards,
                         'sample_point_cnt': sample_point_cnt,
                         'train_set': train_set,
                         'total_train_time': end_time - start_time},
                        open(self.record_dir + '/cumulative_rewards_start_from_seed1576_pret.pik', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pickle.dump({'cumulative_rewards': cumulative_rewards,
                         'sample_point_cnt': sample_point_cnt,
                         'train_set': train_set,
                         'total_train_time': end_time - start_time},
                        open(self.record_dir + '/cumulative_rewards.pik', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

    def rollout_br_policy(self, env, agent_id=0, br=True, store_tuple=True, spec_target=None):
        args = self.args
        episode_reward = 0
        episode_dist = []
        if spec_target is None:
            if args.pol_type == 'hyper' or args.pol_type == 'mlp_plus' or args.pol_type == 'unified' \
                    or args.pol_type == 'simple' or args.pol_type == 'simple2' or args.pol_type == 'simplenohead'\
                    or args.pol_type == 'simplenhll':
                if args.mask_N:
                    meta_v = torch.zeros(args.agent_num, dtype=torch.float32).unsqueeze(1)
                else:
                    if args.meta_v_dim > 1:
                        meta_v = torch.FloatTensor(np.array([self.N_emb[args.agent_num]] * args.agent_num))
                    else:
                        meta_v = torch.FloatTensor(np.array([args.agent_num] * args.agent_num)).unsqueeze(1)
                if args.cuda:
                    meta_v = meta_v.to(args.device)
                if args.norm_N:
                    meta_v = (meta_v - args.min_agent_num) / (args.max_agent_num - args.min_agent_num)
            else:
                meta_v = None
        else:
            if spec_target == 'hyper' or spec_target == 'mlp_plus' or args.pol_type == 'unified' or\
                    args.pol_type == 'simple' or args.pol_type == 'simple2' or args.pol_type == 'simplenohead'\
                    or args.pol_type == 'simplenhll':
                if args.meta_v_dim > 1:
                    meta_v = torch.FloatTensor(np.array([self.N_emb[args.agent_num]] * args.agent_num))
                else:
                    meta_v = torch.tensor([args.agent_num] * args.agent_num, dtype=torch.float32).unsqueeze(1)
                if args.cuda:
                    meta_v = meta_v.to(args.device)
            else:
                meta_v = None
        if args.pos_emb:
            pos_emb = torch.FloatTensor(np.array([self.N_emb_sincos[args.agent_num]] * args.agent_num))
            if args.cuda:
                pos_emb = pos_emb.to(args.device)
        else:
            pos_emb = None
        env.reset()
        episode_dist.append(env.get_agent_dist())
        obs, act_masks = env.get_obs(0)
        dones = [False] * args.agent_num
        for t in range(args.episode_len):
            if spec_target is None:
                if args.pol_type == 'hyper' or args.pol_type == 'mlp_plus' or args.pol_type == 'unified' or\
                        args.pol_type == 'simple' or args.pol_type == 'simple2' or args.pol_type == 'simplenohead'\
                        or args.pol_type == 'simplenhll':
                    if args.fine_tunning:
                        actions = self.policy.select_action_finetune(meta_v, obs, store_tuple=False, pos_emb=pos_emb)
                    else:
                        actions = self.policy.select_action(meta_v, obs, store_tuple=False, pos_emb=pos_emb)
                elif args.pol_type == 'mlp':
                    actions = self.policy.select_action(obs, store_tuple=False)
                else:
                    raise ValueError(f"Unknown policy type {args.pol_type}")
            else:
                if spec_target == 'hyper' or spec_target == 'mlp_plus':
                    actions = self.policy.select_action(meta_v, obs, store_tuple=False)
                elif args.pol_type == 'mlp':
                    actions = self.policy.select_action(obs, store_tuple=False)
                else:
                    raise ValueError(f"Unknown policy type {args.pol_type}")
            if br:
                act = self.br_policy.select_action(obs, store_tuple=store_tuple, store_tuple_idx=agent_id)
                actions[agent_id] = act[agent_id]
            rewards = env.step(t + 1, actions, act_masks)
            obs, act_masks = env.get_obs(t + 1)
            if t + 1 == args.episode_len:
                dones = [True] * args.agent_num
            if br and store_tuple:
                if args.alg == 'ppo':
                    self.br_policy.buffer.rewards.append(rewards[agent_id])
                    self.br_policy.buffer.is_terminals.append(dones[agent_id])
            episode_reward += rewards[agent_id]
            episode_dist.append(env.get_agent_dist())

        return episode_reward, episode_dist

    def exploitability(self, agent_num, writer=None):
        self.args.agent_num = agent_num
        args = self.args
        cuda = args.cuda
        if args.env_name == 'taxi':
            env = TaxiSimulator(args)
        elif args.env_name == 'crowd':
            env = Crowd(args)
        elif args.env_name == 'explore':
            env = Explore2d(args)
        else:
            raise ValueError(f"Not implemented env {args.env_name}")
        agent_id = 0

        #======== file names for storing results =============
        if args.pretrain and args.pretrain_model:
            br_rew_str = '/transfer_agent_num{}_br_episode_reward_from{}{}{}{}_start_from_seed1576_pret.pik' \
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            br_pol_str = 'transfer_agent_num{}_br_policy_from{}{}{}{}_start_from_seed1576_pret.pth' \
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            retrain_rew_str = '/transfer_agent_num{}_retrain_episode_reward_from{}{}{}{}_start_from_seed1576_pret.pik' \
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            retrain_pol_str = 'transfer_agent_num{}_retrain_policy_from{}{}{}{}_start_from_seed1576_pret.pth' \
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            fine_tunning_rew_str = '/transfer_agent_num{}_ft_episode_reward_from{}{}{}{}_start_from_seed1576_pret.pik' \
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            fine_tunning_pol_str = 'transfer_agent_num{}_ft_policy_from{}{}{}{}_start_from_seed1576_pret.pth' \
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
        else:
            br_rew_str = '/transfer_agent_num{}_br_episode_reward_from{}{}{}{}.pik'\
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            br_pol_str = 'transfer_agent_num{}_br_policy_from{}{}{}{}.pth'\
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            retrain_rew_str = '/transfer_agent_num{}_retrain_episode_reward_from{}{}{}{}.pik'\
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            retrain_pol_str = 'transfer_agent_num{}_retrain_policy_from{}{}{}{}.pth'\
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            fine_tunning_rew_str = '/transfer_agent_num{}_ft_episode_reward_from{}{}{}{}.pik' \
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
            fine_tunning_pol_str = 'transfer_agent_num{}_ft_policy_from{}{}{}{}.pth' \
                .format(agent_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)
        #=====================================================

        if args.re_train:
            self.policy.re_init()
            episode_reward = []
            episode_cnt = 0
            if args.pol_type == 'hyper' or args.pol_type == 'mlp_plus' or args.pol_type == 'unified' or \
                    args.pol_type == 'simple' or args.pol_type == 'simple2' or args.pol_type == 'simplenohead'\
                    or args.pol_type == 'simplenhll':
                if args.mask_N:
                    meta_v = torch.zeros(args.agent_num, dtype=torch.float32).unsqueeze(1)
                else:
                    if args.meta_v_dim > 1:
                        meta_v = torch.FloatTensor(np.array([self.N_emb[args.agent_num]] * args.agent_num))
                    else:
                        meta_v = torch.FloatTensor(np.array([args.agent_num] * args.agent_num)).unsqueeze(1)
                if args.norm_N:
                    meta_v = (meta_v - args.min_agent_num) / (args.max_agent_num - args.min_agent_num)
                if cuda:
                    meta_v = meta_v.to(args.device)
            else:
                meta_v = None
            if args.pos_emb:
                pos_emb = torch.FloatTensor(np.array([self.N_emb_sincos[args.agent_num]] * args.agent_num))
                if args.cuda:
                    pos_emb = pos_emb.to(args.device)
            else:
                pos_emb = None
            while episode_cnt < args.re_train_episodes:
                start = datetime.now().timestamp()
                reward, _ = self.rollout(meta_v, env, pos_emb=pos_emb)
                episode_reward.append(reward)
                if writer is not None:
                    writer.add_scalar('retrain_reward_agent_num{}'.format(agent_num), reward, episode_cnt)
                if args.alg == 'ppo':
                    if (episode_cnt + 1) % self.args.update_episodes == 0:
                        self.policy.update()
                end = datetime.now().timestamp()
                episode_time = (end - start)  # second
                if episode_cnt % args.print_episodes == 0 or episode_cnt == args.re_train_episodes - 1:
                    print("(Exploita-Retrain) Seed:{}, env:{}{}, Epi:#{}/{}, AvgR:{:.4f}, Pol:{}-{}, Tran N:{}, MP:{}, T:{:.3f}"
                          .format(args.seed, args.env_name, self.map_type_str, episode_cnt, args.re_train_episodes, reward,
                                  args.pol_type, args.alg, args.agent_num, self.multi_point_str, episode_time))
                if (episode_cnt + 1) % 1000 == 0:
                    pickle.dump({'episode_reward': episode_reward}, open(self.record_dir + retrain_rew_str, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                episode_cnt += 1
            self.policy.save("".join([self.checkpoint_dir, retrain_pol_str]))

        if args.fine_tunning:
            episode_reward = []
            episode_cnt = 0
            if args.pol_type == 'hyper' or args.pol_type == 'mlp_plus' or args.pol_type == 'unified' or \
                    args.pol_type == 'simple' or args.pol_type == 'simple2' or args.pol_type == 'simplenohead'\
                    or args.pol_type == 'simplenhll':
                if args.mask_N:
                    meta_v = torch.zeros(args.agent_num, dtype=torch.float32).unsqueeze(1)
                else:
                    if args.noise_N:
                        meta_v = torch.FloatTensor(args.agent_num - 0.5 + np.random.rand(args.agent_num)).unsqueeze(1)
                    else:
                        if args.meta_v_dim > 1:
                            meta_v = torch.FloatTensor(np.array([self.N_emb[args.agent_num]] * args.agent_num))
                        else:
                            meta_v = torch.FloatTensor(np.array([args.agent_num] * args.agent_num)).unsqueeze(1)
                if args.norm_N:
                    meta_v = (meta_v - args.min_agent_num) / (args.max_agent_num - args.min_agent_num)
                if cuda:
                    meta_v = meta_v.to(args.device)
            else:
                meta_v = None
            if args.pos_emb:
                pos_emb = torch.FloatTensor(np.array([self.N_emb_sincos[args.agent_num]] * args.agent_num))
                if args.cuda:
                    pos_emb = pos_emb.to(args.device)
            else:
                pos_emb = None
            self.policy.init_policy_finetune(meta_v[0])
            while episode_cnt < args.fine_tunning_episodes:
                start = datetime.now().timestamp()
                # if args.pol_type == 'hyper' or args.pol_type == 'mlp_plus' or args.pol_type == 'unified' or \
                #         args.pol_type == 'simple' or args.pol_type == 'simple2' or args.pol_type == 'simplenohead'\
                #         or args.pol_type == 'simplenhll':
                #     if args.meta_v_dim > 1:
                #         N_emb = self.N_emb[args.agent_num]
                #         meta_v = torch.FloatTensor(np.array([N_emb] * args.agent_num))
                #         if cuda:
                #             meta_v = meta_v.to(args.device)
                reward, _ = self.rollout(meta_v, env, pos_emb=pos_emb, finetune=True)
                episode_reward.append(reward)
                if writer is not None:
                    writer.add_scalar('fine_tunning_reward_agent_num{}'.format(agent_num), reward, episode_cnt)
                if args.alg == 'ppo':
                    if (episode_cnt + 1) % self.args.update_episodes == 0:
                        self.policy.update_finetune()
                end = datetime.now().timestamp()
                episode_time = (end - start)  # second
                if episode_cnt % args.print_episodes == 0 or episode_cnt == args.fine_tunning_episodes - 1:
                    print("(Fine-Tune) Seed:{}, env:{}{}, Epi:#{}/{}, AvgR:{:.4f}, Pol:{}-{}, Tran N:{}, MP:{}, T:{:.3f}"
                          .format(args.seed, args.env_name, self.map_type_str, episode_cnt, args.fine_tunning_episodes,
                                  reward, args.pol_type, args.alg, args.agent_num, self.multi_point_str, episode_time))
                if (episode_cnt + 1) % 1000 == 0:
                    pickle.dump({'episode_reward': episode_reward}, open(self.record_dir + fine_tunning_rew_str, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
                episode_cnt += 1
            self.policy.save_finetune("".join([self.checkpoint_dir, fine_tunning_pol_str]))

        spec_target = None
        episode_cnt = 0
        if args.transfer_load_br_checkpoint:
            episode_cnt = args.transfer_load_br_checkpoint_step
        episode_reward = []
        while episode_cnt < args.transfer_br_max_episodes:
            start = datetime.now().timestamp()
            reward, _ = self.rollout_br_policy(env, agent_id, True, True, spec_target)
            if writer is not None:
                writer.add_scalar('br_reward_agent_num{}'.format(agent_num), reward, episode_cnt)
            episode_reward.append(reward)
            if args.alg == 'ppo':
                if (episode_cnt + 1) % args.update_episodes == 0:
                    self.br_policy.update()
            end = datetime.now().timestamp()
            episode_time = (end - start)  # second
            if episode_cnt % args.print_episodes == 0 or episode_cnt == args.transfer_br_max_episodes - 1:
                print("(Expolita) Seed:{}, env:{}{}, Epi:#{}/{}, BR:{:.4f}, Pol:{}-{}, Tran N:{}, MP:{}, T:{:.3f}, {}"
                      .format(args.seed, args.env_name, self.map_type_str, episode_cnt, args.transfer_br_max_episodes, reward,
                              args.pol_type, args.alg, args.agent_num, self.multi_point_str, episode_time, self.fine_tunning_str))
            if (episode_cnt + 1) % 1000 == 0:
                pickle.dump({'episode_reward': episode_reward}, open(self.record_dir + br_rew_str, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
            episode_cnt += 1
        self.br_policy.save("".join([self.checkpoint_dir, br_pol_str]))

        nash_conv = 0.
        epi_nash_conv = 1000
        for epi in range(epi_nash_conv):
            reward_cur_pol, _ = self.rollout_br_policy(env, agent_id, False, False, spec_target)
            reward_bre_pol, _ = self.rollout_br_policy(env, agent_id, True, False, spec_target)
            nc = max(0., reward_bre_pol - reward_cur_pol)
            nash_conv += nc
            if epi % args.print_episodes == 0 or epi == epi_nash_conv - 1:
                print("(Expolita) Seed:{}, env:{}{}, Epi:#{}/{}, CR:{:.4f}, BR:{:.4f}, Expol:{:.4f}, Pol:{}-{}, Tran N:{}, MP:{}, {}"
                      .format(args.seed, args.env_name, self.map_type_str, epi, epi_nash_conv, reward_cur_pol, reward_bre_pol,
                              nc, args.pol_type, args.alg, args.agent_num, self.multi_point_str, self.fine_tunning_str))
        nash_conv /= epi_nash_conv

        return nash_conv

    def transfer(self):
        args = self.args
        if args.pretrain and args.pretrain_model:
            pol_str = "policy_{}_start_from_seed1576_pret.pth".format(args.transfer_from_episodes)
        else:
            pol_str = "policy_{}.pth".format(args.transfer_from_episodes)
        print('Load current policy from episodes {}, policy path: {}'.format(args.transfer_from_episodes, pol_str))
        self.policy.load(self.record_dir + '/checkpoints/' + pol_str)

        self.retrain_str = ""
        if args.re_train:
            self.retrain_str = "_fix_c{}_retrain_step{}".format(int(args.fix_critic), args.re_train_episodes)
        self.br_str = "_br_step{}".format(args.transfer_br_max_episodes)
        self.fine_tunning_str = ""
        if args.fine_tunning:
            self.fine_tunning_str = "_ft{}_step{}"\
                .format(int(args.fine_tunning), args.fine_tunning_episodes)

        # transfer_agent_num_list = [40, 45, 50]
        # transfer_agent_num_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        transfer_agent_num_list = [110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

        # if args.wait:
        #     print('Start:  ', datetime.now().replace(microsecond=0))
        #     print('Wait(s):', args.wait_time)
        #     print('N List: ', transfer_agent_num_list)
        #     print(self.run_name)
        #     time.sleep(args.wait_time)

        start_time = datetime.now().replace(microsecond=0)
        for transfer_num in transfer_agent_num_list:
            if args.transfer_load_br_checkpoint:
                br_pol_checkpoint_str = 'transfer_agent_num{}_br_policy_from2000000_br_step{}_start_from_seed1576_pret.pth'\
                    .format(transfer_num, args.transfer_load_br_checkpoint_step)
                print('Load saved BR policy: {}'.format(br_pol_checkpoint_str))
                self.br_policy.load("".join([self.checkpoint_dir, br_pol_checkpoint_str]))
            exploitability = self.exploitability(transfer_num, self.writer)
            print("(Expolita) Seed:{}, env:{}{}, Expolita:{:.4f}, Pol:{}-{}, Tran N:{}"
                  .format(args.seed, args.env_name, self.map_type_str, exploitability, args.pol_type, args.alg, transfer_num))

            if args.pretrain and args.pretrain_model:
                res_file_str = '/transfer_agent_num{}_exploitability_from{}{}{}{}_start_from_seed1576_pret.pik' \
                    .format(transfer_num, args.transfer_from_episodes, self.br_str, self.retrain_str,
                            self.fine_tunning_str)
            else:
                res_file_str = '/transfer_agent_num{}_exploitability_from{}{}{}{}.pik'\
                .format(transfer_num, args.transfer_from_episodes, self.br_str, self.retrain_str, self.fine_tunning_str)

            pickle.dump({'exploitability': exploitability}, open(self.record_dir + res_file_str, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)
        print("Finished training at (GMT) : ", end_time)
        print("Total training time  : ", end_time - start_time)
        print("============================================================================================")

    def eval(self, init_checkpoint):
        args = self.args
        cuda = args.cuda

        print('Load {} checkpoint {}: {}'.format(args.alg, init_checkpoint, self.args.checkpoint_dir + '/policy_' + str(init_checkpoint) + '.pth'))
        self.policy.load(self.args.checkpoint_dir + '/policy_' + str(init_checkpoint) + '.pth')
        start_episode = 0
    
        start_time = datetime.now().replace(microsecond=0)

        episode_cnt = start_episode
        cumulative_rewards = []
        sample_point_cnt = np.zeros(args.max_agent_num + 1)
        if args.train_set:
            train_set = sample_unseen_task(args.min_agent_num, args.max_agent_num + 1, num_of_unseen=args.num_train_task, method=args.mixenv_dist, lam=args.lam)
        else:
            train_set = []

        while episode_cnt < args.max_episodes:
            start = datetime.now().timestamp()
            if args.min_agent_num != args.max_agent_num:
                # sample a new env
                if episode_cnt % args.sample_inter == 0:
                    if args.train_set:
                        np.random.seed(episode_cnt)
                        a_num = np.random.choice(train_set)
                    else:
                        if args.mixenv_dist == 'uniform':
                            np.random.seed(episode_cnt)
                            a_num = np.random.randint(args.min_agent_num, args.max_agent_num + 1)
                        elif args.mixenv_dist == 'poisson':
                            np.random.seed(episode_cnt)
                            while True:
                                a_num = np.random.poisson(args.lam, size=1)
                                if args.min_agent_num <= a_num <= args.max_agent_num:
                                    break
                        elif args.mixenv_dist == 'exponent':
                            np.random.seed(episode_cnt)
                            while True:
                                a_num = int(np.round(np.random.exponential(args.lam, size=1)))
                                if args.min_agent_num <= a_num <= args.max_agent_num:
                                    break
                        elif args.mixenv_dist == 'seq':
                            if a_num < args.max_agent_num:
                                a_num += 1
                            else:
                                a_num = args.min_agent_num
                        else:
                            raise ValueError(f"Unknown env sample method: {args.mexenv_dist}")
                sample_point_cnt[int(a_num)] += 1
                self.env.reset_agent_pos(a_num)
            cur_agent_num = self.env.agent_num

            meta_v = None
            pos_emb = None

            reward, _ = self.rollout(meta_v, pos_emb=pos_emb, store_tuple=False)
            cumulative_rewards.append(reward)
            
            self.writer.add_scalar('reward', reward, episode_cnt)
           
            end = datetime.now().timestamp()
            episode_time = (end - start)  # second

            if episode_cnt % args.print_episodes == 0 or episode_cnt == args.max_episodes - 1:
                print("(Eval) Seed:{}, env:{}{}, Epi:#{}/{}, AvgR:{:.4f}, Pol:{}-{}, N:{}, MP:{}, T:{:.3f}"
                      .format(args.seed, args.env_name, self.map_type_str, episode_cnt, args.max_episodes, reward,
                              args.pol_type, args.alg, cur_agent_num, self.multi_point_str, episode_time))

            episode_cnt += 1

        print("============================================================================================")
        end_time = datetime.now().replace(microsecond=0)
        print("Started eval at (GMT) : ", start_time)
        print("Finished eval at (GMT) : ", end_time)
        print("Total eval time  : ", end_time - start_time)
        print("============================================================================================")
