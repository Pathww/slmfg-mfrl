import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .hyper_nets import HyperNetwork
from utils.utils import init_weights_kaiming_uniform, l1_reg_loss, get_1d_sincos_pos_embed_from_grid


class RolloutBuffer:
    def __init__(self):
        self.meta_vs = []
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.meta_vs[:]
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class HyperA(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim=256):
        super(HyperA, self).__init__()

        self.actor = HyperNetwork(meta_v_dim, state_dim, action_dim, hidden_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.critic.apply(init_weights_kaiming_uniform)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class HyperC(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim=256):
        super(HyperC, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor.apply(init_weights_kaiming_uniform)
        self.critic = HyperNetwork(meta_v_dim, state_dim, 1, hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        self.actor.apply(init_weights_kaiming_uniform)

    def act(self, meta_v, state):
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action):
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)

        return action_logprobs, state_values, dist_entropy


class HyperAC(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim=256):
        super(HyperAC, self).__init__()

        self.actor = HyperNetwork(meta_v_dim, state_dim, action_dim, hidden_dim)
        self.critic = HyperNetwork(meta_v_dim, state_dim, 1, hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False):
        action_probs, _ = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()

        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, actor_get_layer_out=False, actor_layer=None, critic_get_layer_out=False, critic_layer=None):
        action_probs, actor_layer_out = self.actor(meta_v, state, get_layer_out=actor_get_layer_out, layer=actor_layer)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values, critic_layer_out = self.critic(meta_v, state, get_layer_out=critic_get_layer_out, layer=critic_layer)

        return action_logprobs, state_values, dist_entropy, actor_layer_out, critic_layer_out

    def get_dynamic_weights(self, meta_v):
        return self.actor.get_weight(meta_v)


class HyperPPO:
    def __init__(self, meta_v_dim, state_dim, action_dim, lr_a, lr_c, gamma, K_epochs, eps_clip, device, cuda,
                 min_meta_v, max_meta_v, pos_emb, pos_emb_dim, hidden_dim=256, hyper_actor=False, hyper_critic=True, reg_actor=False,
                 reg_actor_layer=None, reg_critic=False, reg_critic_layer=None, fix_critic=False, optimizer='adam',
                 w_decay=0, l1=False, w_clip=False, w_clip_p=10, max_pe_encode=10000):
        self.meta_v_dim = meta_v_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.device = device
        self.cuda = cuda
        self.hyper_actor = hyper_actor
        self.hyper_critic = hyper_critic
        self.min_meta_v = min_meta_v
        self.max_meta_v = max_meta_v
        self.reg_actor = reg_actor
        self.reg_actor_layer = reg_actor_layer
        self.reg_critic = reg_critic
        self.reg_critic_layer = reg_critic_layer
        self.fix_critic = fix_critic
        self.l1 = l1
        self.w_clip = w_clip
        self.w_clip_p = w_clip_p
        self.pos_emb = pos_emb
        self.pos_emb_dim = pos_emb_dim

        self.buffer = RolloutBuffer()

        meta_input_dim = meta_v_dim if not pos_emb else pos_emb_dim

        if hyper_actor and hyper_critic:
            self.policy = HyperAC(meta_input_dim, state_dim, action_dim, hidden_dim)
            self.policy_old = HyperAC(meta_input_dim, state_dim, action_dim, hidden_dim)
        else:
            if hyper_actor:
                self.policy = HyperA(meta_input_dim, state_dim, action_dim, hidden_dim)
                self.policy_old = HyperA(meta_input_dim, state_dim, action_dim, hidden_dim)
            elif hyper_critic:
                self.policy = HyperC(meta_input_dim, state_dim, action_dim, hidden_dim)
                self.policy_old = HyperC(meta_input_dim, state_dim, action_dim, hidden_dim)
            else:
                raise ValueError(f'Hypernet should be specified for actor or/and critic, '
                                 f'but {hyper_actor} for actor and {hyper_critic} for critic.')
        if fix_critic:
            if optimizer == 'adam':
                self.optimizer = torch.optim.Adam([
                    {'params': self.policy.actor.parameters(), 'lr': lr_a}
                ], weight_decay=w_decay)
            elif optimizer == 'adamw':
                self.optimizer = torch.optim.AdamW([
                    {'params': self.policy.actor.parameters(), 'lr': lr_a}
                ], weight_decay=w_decay)
            else:
                raise ValueError(f'Unknown optimizer {optimizer}')
        else:
            if optimizer == 'adam':
                self.optimizer = torch.optim.Adam([
                    {'params': self.policy.actor.parameters(), 'lr': lr_a},
                    {'params': self.policy.critic.parameters(), 'lr': lr_c}
                ], weight_decay=w_decay)
            elif optimizer == 'adamw':
                self.optimizer = torch.optim.AdamW([
                    {'params': self.policy.actor.parameters(), 'lr': lr_a},
                    {'params': self.policy.critic.parameters(), 'lr': lr_c}
                ], weight_decay=w_decay)
            else:
                raise ValueError(f'Unknown optimizer {optimizer}')

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')  # note that use batchmean

        if cuda:
            self.policy.to(device)
            self.policy_old.to(device)
            self.MseLoss.to(device)
            self.KLDivLoss.to(device)

        self.N_emb = np.array([[int(x) for x in '{0:012b}'.format(i)] for i in range(4096)])
        self.N_emb_sincos = get_1d_sincos_pos_embed_from_grid(pos_emb_dim, np.arange(0, max_meta_v + 1), max_pe_encode)

        # model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('Hyper: ', params)
        # input('Count Hyper')

    def pretrain(self, min_meta_v, max_meta_v, ppo, pretrain_batch_size, pretrain_step, lr_pretrain,
                 writer, seed, env, episode_len, norm_N, loss_type='mse', pret_actor=True, pret_critic=True, info_str=None):
        if pret_actor and pret_critic and self.hyper_actor and self.hyper_critic:
            optimizer = torch.optim.Adam([
                {'params': self.policy.actor.parameters(), 'lr': lr_pretrain},
                {'params': self.policy.critic.parameters(), 'lr': lr_pretrain}
            ])
        else:
            if pret_actor and self.hyper_actor:
                optimizer = torch.optim.Adam([
                    {'params': self.policy.actor.parameters(), 'lr': lr_pretrain}
                ])
            elif pret_critic and self.hyper_critic:
                optimizer = torch.optim.Adam([
                    {'params': self.policy.critic.parameters(), 'lr': lr_pretrain}
                ])
            else:
                raise ValueError(f'Specification does not match: pretrain_actor {pret_actor}/hyper_actor {self.hyper_actor},'
                                 f' pretrain_critic {pret_critic}/hyper_critic {self.hyper_critic}.')
        np.random.seed(seed)
        for i in range(pretrain_step):
            if self.pos_emb:
                rand_N = np.random.randint(self.min_meta_v, self.max_meta_v + 1, pretrain_batch_size)
                rand_N_emb = self.N_emb_sincos[rand_N, :]
                meta_v_tensor = torch.FloatTensor(rand_N_emb)
            else:
                if self.meta_v_dim > 1:
                    rand_N = np.random.randint(self.min_meta_v, self.max_meta_v + 1, pretrain_batch_size)
                    rand_N_emb = self.N_emb[rand_N, :]
                    meta_v_tensor = torch.FloatTensor(rand_N_emb)
                else:
                    meta_v_tensor = torch.randint(self.min_meta_v, self.max_meta_v + 1, [pretrain_batch_size, 1], dtype=torch.float32)
                    if norm_N:
                        meta_v_tensor = (meta_v_tensor - min_meta_v) / (max_meta_v - min_meta_v)
            # meta_v_tensor = min_meta_v + (max_meta_v - min_meta_v) * torch.rand([pretrain_batch_size, 1])
            # meta_v_tensor = torch.FloatTensor(meta_v_tensor)
            if self.cuda:
                meta_v_tensor = meta_v_tensor.to(self.device)
            cur_t = np.eye(episode_len + 1)[np.random.randint(0, episode_len + 1, pretrain_batch_size)]
            if env.env_name == 'taxi':
                x = np.eye(env.args.map_M)[np.random.randint(0, env.args.map_M, pretrain_batch_size)]
                y = np.eye(env.args.map_N)[np.random.randint(0, env.args.map_N, pretrain_batch_size)]
                samples = np.concatenate((x, y, cur_t), 1)
            elif env.env_name == 'crowd2d' or env.env_name == 'crowd2dv1' or env.env_name == 'crowd2dv3' \
                    or env.env_name == 'crowd2dv4' or env.env_name == 'crowd2dv5' or env.env_name == 'crowd2dv6':
                x = np.eye(env.map_M)[np.random.randint(0, env.map_M, pretrain_batch_size)]
                y = np.eye(env.map_N)[np.random.randint(0, env.map_N, pretrain_batch_size)]
                samples = np.concatenate((x, y, cur_t), 1)
            elif env.env_name == 'crowd' or env.env_name == 'crowdv1'or env.env_name == 'crowdv2':
                x = np.eye(env.map_M)[np.random.randint(0, env.map_M, pretrain_batch_size)]
                samples = np.concatenate((x, cur_t), 1)
            elif env.env_name == 'lq':
                x = np.eye(env.map_M)[np.random.randint(0, env.map_M, pretrain_batch_size)]
                samples = np.concatenate((x, cur_t), 1)
            elif env.env_name == 'sis':
                x = np.eye(env.map_M)[np.random.randint(0, env.map_M, pretrain_batch_size)]
                samples = np.concatenate((x, cur_t), 1)
            elif env.env_name == 'explore' or env.env_name == 'explorev1':
                x = np.eye(env.map_M)[np.random.randint(0, env.map_M, pretrain_batch_size)]
                y = np.eye(env.map_N)[np.random.randint(0, env.map_N, pretrain_batch_size)]
                samples = np.concatenate((x, y, cur_t), 1)
            elif env.env_name == 'evacuate' or env.env_name == 'evacuatev1' or env.env_name == 'evacuatev2':
                f = np.eye(env.floor_num)[np.random.randint(0, env.floor_num, pretrain_batch_size)]
                x = np.eye(env.map_M)[np.random.randint(0, env.map_M, pretrain_batch_size)]
                y = np.eye(env.map_N)[np.random.randint(0, env.map_N, pretrain_batch_size)]
                samples = np.concatenate((f, x, y, cur_t), 1)
            elif env.env_name == 'invest':
                x = np.random.rand(pretrain_batch_size, env.map_M)
                samples = np.concatenate((x, cur_t), 1)
            elif env.env_name == 'malware':
                x = np.random.rand(pretrain_batch_size, env.map_M)
                samples = np.concatenate((x, cur_t), 1)
            else:
                raise ValueError(f'Unknown env {env.env_name}')
            # samples = torch.rand([pretrain_batch_size, self.state_dim])
            samples = torch.FloatTensor(samples)
            if self.cuda:
                samples = samples.to(self.device)
            if pret_actor and pret_critic and self.hyper_actor and self.hyper_critic:
                if loss_type == 'kl':
                    with torch.no_grad():
                        label_a = F.softmax(ppo.policy.actor(samples), dim=1)
                        label_c = F.softmax(ppo.policy.critic(samples), dim=1)
                    out_a = torch.log_softmax(self.policy.actor(meta_v_tensor, samples), dim=1)
                    out_c = torch.log_softmax(self.policy.critic(meta_v_tensor, samples), dim=1)
                    loss = self.KLDivLoss(out_a, label_a) + self.KLDivLoss(out_c, label_c)
                elif loss_type == 'mse':
                    with torch.no_grad():
                        label_a = ppo.policy.actor(samples)
                        label_c = ppo.policy.critic(samples)
                    out_a, _ = self.policy.actor(meta_v_tensor, samples)
                    out_c, _ = self.policy.critic(meta_v_tensor, samples)
                    loss = F.mse_loss(out_a, label_a) + F.mse_loss(out_c, label_c)
                else:
                    raise ValueError(f'Unknown loss type {loss_type}')
            else:
                if pret_actor and self.hyper_actor:
                    if loss_type == 'kl':
                        with torch.no_grad():
                            label_a = F.softmax(ppo.policy.actor(samples), dim=1)
                        out_a = torch.log_softmax(self.policy.actor(meta_v_tensor, samples), dim=1)
                        loss = self.KLDivLoss(out_a, label_a)
                    elif loss_type == 'mse':
                        with torch.no_grad():
                            label_a = ppo.policy.actor(samples)
                        out_a = self.policy.actor(meta_v_tensor, samples)
                        loss = F.mse_loss(out_a, label_a)
                    else:
                        raise ValueError(f'Unknown loss type {loss_type}')
                elif pret_critic and self.hyper_critic:
                    if loss_type == 'kl':
                        with torch.no_grad():
                            label_c = F.softmax(ppo.policy.critic(samples), dim=1)
                        out_c = torch.log_softmax(self.policy.critic(meta_v_tensor, samples), dim=1)
                        loss = self.KLDivLoss(out_c, label_c)
                    elif loss_type == 'mse':
                        with torch.no_grad():
                            label_c = ppo.policy.critic(samples)
                        out_c = self.policy.critic(meta_v_tensor, samples)
                        loss = F.mse_loss(out_c, label_c)
                    else:
                        raise ValueError(f'Unknown loss type {loss_type}')
                else:
                    raise ValueError(
                        f'Specification does not match: pretrain_actor {pret_actor}/hyper_actor {self.hyper_actor},'
                        f' pretrain_critic {pret_critic}/hyper_critic {self.hyper_critic}.')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('pretrain_loss', loss.detach().item(), i)
            if i % 1000 == 0:
                print("(Pret)Seed:{}, env:{}, Step:#{}/{}, Loss:{:.6f}, Type:{}, Pol:{}, N:{}-{}, Info:{}"
                      .format(seed, env.env_name, i, pretrain_step, loss.detach().item(), loss_type,
                              'hyper_a{}_c{}'.format(int(self.hyper_actor), int(self.hyper_critic)),
                              min_meta_v, max_meta_v, 0 if info_str is None else info_str))
            if loss.detach().item() < 1e-6:
                break
        self.policy_old.load_state_dict(self.policy.state_dict())

    def get_weights(self, meta_v):
        return self.policy.actor.get_weight(meta_v)

    def select_action(self, meta_v, state, store_tuple=True, store_tuple_idx=0, ret_prob=False, pos_emb=None):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            if self.cuda:
                state = state.to(self.device)
            if ret_prob:
                action, action_logprob, action_prob = self.policy_old.act(meta_v, state, ret_prob)
            else:
                action, action_logprob = self.policy_old.act(meta_v, state)
        if store_tuple:
            self.buffer.meta_vs.append(meta_v[store_tuple_idx])
            self.buffer.states.append(state[store_tuple_idx])
            self.buffer.actions.append(action[store_tuple_idx])
            self.buffer.logprobs.append(action_logprob[store_tuple_idx])

        if ret_prob:
            return action.detach().cpu().numpy(), action_prob.detach().cpu().numpy()

        return action.detach().cpu().numpy()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        if self.cuda:
            rewards = rewards.to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_meta_vs = torch.squeeze(torch.stack(self.buffer.meta_vs, dim=0)).detach()
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        if self.cuda:
            old_meta_vs = old_meta_vs.to(self.device)
            old_states = old_states.to(self.device)
            old_actions = old_actions.to(self.device)
            old_logprobs = old_logprobs.to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy, actor_layer_out_o, critic_layer_out_o \
                = self.policy.evaluate(old_meta_vs, old_states, old_actions,
                                       actor_get_layer_out=self.reg_actor, actor_layer=self.reg_actor_layer,
                                       critic_get_layer_out=self.reg_critic, critic_layer=self.reg_critic_layer)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            if self.fix_critic:
                loss = -torch.min(surr1, surr2) - 0.01 * dist_entropy
            else:
                # loss_value = self.MseLoss(state_values, rewards)
                # loss_actor = -torch.min(surr1, surr2).mean()
                # loss_entropy = 0.01 * dist_entropy.mean()
                # loss_total = (-torch.min(surr1, surr2) - 0.01 * dist_entropy).mean() + 0.5 * self.MseLoss(state_values, rewards)
                loss = (-torch.min(surr1, surr2) - 0.01 * dist_entropy).mean() + 0.5 * self.MseLoss(state_values, rewards)

                # loss for regularizer
                if self.reg_actor or self.reg_critic:
                    if self.meta_v_dim > 1:
                        rand_N = np.random.randint(self.min_meta_v, self.max_meta_v + 1, len(old_meta_vs))
                        rand_N_emb = self.N_emb[rand_N, :]
                        neg_N = torch.FloatTensor(rand_N_emb)
                    else:
                        neg_N = torch.randint(self.min_meta_v, self.max_meta_v + 1, old_meta_vs.shape, dtype=torch.float32)
                    if self.cuda:
                        neg_N = neg_N.to(self.device)
                    if self.reg_actor:
                        _, _, _, actor_layer_out_n, _ \
                            = self.policy.evaluate(neg_N.unsqueeze(1), old_states, old_actions,
                                                   actor_get_layer_out=self.reg_actor, actor_layer=self.reg_actor_layer,
                                                   critic_get_layer_out=self.reg_critic, critic_layer=self.reg_critic_layer)
                        loss = loss + 0.1 * self.MseLoss(actor_layer_out_o, actor_layer_out_n)
                    if self.reg_critic:
                        _, _, _, _, critic_layer_out_n \
                            = self.policy.evaluate(neg_N.unsqueeze(1), old_states, old_actions,
                                                   actor_get_layer_out=self.reg_actor, actor_layer=self.reg_actor_layer,
                                                   critic_get_layer_out=self.reg_critic, critic_layer=self.reg_critic_layer)
                        loss = loss + 0.1 * self.MseLoss(critic_layer_out_o, critic_layer_out_n)

                if self.l1: # l1 regularization
                    # l1_loss = l1_reg_loss(self.policy)
                    # print('main loss: {:.8f}, l1 loss: {:.8f}'.format(loss.item(), l1_loss.item())) # main loss: 0.01~0.2, l1 loss: 40000~50000
                    loss = loss + 1e-7 * l1_reg_loss(self.policy)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            if self.w_clip:  # Weight Clip
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.w_clip_p)
            self.optimizer.step()
            # print(self.policy.actor.input_layer.W1.weight.grad)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def re_init(self):
        self.policy_old.init_actor()
        self.policy.load_state_dict(self.policy_old.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))