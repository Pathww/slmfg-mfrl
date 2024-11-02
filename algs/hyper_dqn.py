import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .hyper_nets import HyperNetwork
from utils.utils import to_Cuda


class ReplayBuffer(object):
    def __init__(self, meta_v_dim, state_dim, act_dim, buffer_size):
        self.meta_v_dim = meta_v_dim
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.buffer_size = buffer_size

        self.states = np.zeros((buffer_size,) + (state_dim, )).astype('float32')
        self.meta_vs = np.zeros((buffer_size,) + (meta_v_dim, )).astype('float32')
        self.states_next = np.zeros((buffer_size,) + (state_dim, )).astype('float32')
        self.actions = np.zeros((buffer_size,) + ()).astype('int32')
        self.rewards = np.zeros((buffer_size,) + ()).astype('float32')
        self.is_terminals = np.zeros((buffer_size,) + ()).astype('float32')
        self.length = 0
        self.flag = 0

    def append_state(self, state):
        self.states[self.flag] = state

    def append_meta_v(self, meta_v):
        self.meta_vs[self.flag] = meta_v

    def append_state_next(self, state_next):
        self.states_next[self.flag] = state_next

    def append_action(self, action):
        self.actions[self.flag] = action

    def append_reward(self, reward):
        self.rewards[self.flag] = reward

    def append_is_terminal(self, is_terminal):
        self.is_terminals[self.flag] = is_terminal

    def add_cnt(self, num=1):
        if self.flag + num >= self.buffer_size:
            self.flag = 0
        self.flag += num
        self.length = min(self.length + num, self.buffer_size)

    def sample(self, batch_size):
        idx = np.random.choice(self.length, size=batch_size, replace=False)
        states = self.states[idx]
        meta_vs = self.meta_vs[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        is_terminals = self.is_terminals[idx]
        states_next = self.states_next[idx]
        return meta_vs, states, actions, rewards, states_next, is_terminals


class ValueNet(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim):
        super(ValueNet, self).__init__()

        self.actor = HyperNetwork(meta_v_dim, state_dim, action_dim, hidden_dim)

    def forward(self, meta_v, state):
        action_values = self.actor(meta_v, state)
        return action_values

    def init_actor(self):
        pass


class HyperDQN:
    def __init__(self, meta_v_dim, state_dim, action_dim, lr, gamma, init_eps, final_eps, eps_decay_step, device, cuda,
                 hidden_dim=256, buffer_size=int(1e6), batch_size=64, update_target_freq=100):
        self.meta_v_dim = meta_v_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.init_eps = init_eps
        self.final_eps = final_eps
        self.eps_decay_step = eps_decay_step
        self.device = device
        self.cuda = cuda
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        self.action_idx = np.array([i for i in range(action_dim)]).astype('int32')
        self.step = 0
        self.buffer = ReplayBuffer(meta_v_dim, state_dim, action_dim, buffer_size)

        self.policy = ValueNet(meta_v_dim, state_dim, action_dim, hidden_dim)
        self.policy_old = ValueNet(meta_v_dim, state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr}
        ])
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MSE_loss = nn.MSELoss()
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')  # note that use batchmean

        if cuda:
            self.policy.to(device)
            self.policy_old.to(device)
            self.MSE_loss.to(device)
            self.KLDivLoss.to(device)

        # model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('Hyper: ', params)
        # input('Count Hyper')

    def pretrain(self, min_meta_v, max_meta_v, dqn, pretrain_batch_size, pretrain_step, lr_pretrain,
                 writer, seed, env_name, loss_type='mse', pret_actor=True, pret_critic=True, info_str=None):
        optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_pretrain}
        ])
        for i in range(pretrain_step):
            meta_v_tensor = torch.rand([pretrain_batch_size, 1])
            meta_v_tensor = torch.FloatTensor(meta_v_tensor)
            if self.cuda:
                meta_v_tensor = meta_v_tensor.to(self.device)
            samples = torch.rand([pretrain_batch_size, self.state_dim])
            samples = torch.FloatTensor(samples)
            if self.cuda:
                samples = samples.to(self.device)
            if loss_type == 'kl':
                with torch.no_grad():
                    label_a = F.softmax(dqn.policy.actor(samples), dim=1)
                out_a = torch.log_softmax(self.policy.actor(meta_v_tensor, samples), dim=1)
                loss = self.KLDivLoss(out_a, label_a)
            elif loss_type == 'mse':
                with torch.no_grad():
                    label_a = dqn.policy.actor(samples)
                out_a = self.policy.actor(meta_v_tensor, samples)
                loss = F.mse_loss(out_a, label_a)
            else:
                raise ValueError(f'Unknown loss type {loss_type}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('pretrain_loss', loss.detach().item(), i)
            if i % 1000 == 0:
                print("(Pret) Seed: {}  env: {}  Step: #{}/{}  Loss: {:.8f}  Type: {}  Policy: {}  Agent N: {}-{}  Info: {}"
                      .format(seed, env_name, i, pretrain_step, loss.detach().item(), loss_type, 'hyper_dqn',
                              min_meta_v, max_meta_v, 0 if info_str is None else info_str))
            if loss.detach().item() < 1e-4:
                break
        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, meta_vs, states, store_tuple=True, store_tuple_idx=0):
        if store_tuple:
            self.buffer.append_state(states[store_tuple_idx])
        n = len(states)
        states = torch.FloatTensor(states)
        if self.cuda:
            states = states.to(self.device)
        action_values = self.policy(meta_vs, states).detach()
        if self.cuda: action_values = action_values.cpu()
        best_actions = torch.argmax(action_values, dim=1).numpy()
        if store_tuple and self.get_epsilon() > 0.0:
            random_actions = np.random.choice(self.action_idx, size=n)
            cond = np.random.uniform(0, 1, size=(n,)) < self.get_epsilon()
            actions = np.where(cond, random_actions, best_actions).astype(np.int32, copy=False)
        else:
            actions = best_actions.astype(np.int32, copy=False)
        if n == 1:
            actions = actions[0]
        if store_tuple:
            if self.cuda: meta_vs = meta_vs.cpu().numpy()
            self.buffer.append_meta_v(meta_vs[store_tuple_idx])
            self.buffer.append_action(actions[store_tuple_idx])

        return actions

    def get_epsilon(self):
        if self.init_eps == self.final_eps:
            return self.init_eps
        if self.eps_decay_step == 0:
            return 0.0
        if self.step > self.eps_decay_step:
            return self.final_eps
        else:
            inter = (self.init_eps - self.final_eps) / self.eps_decay_step
            return self.init_eps - inter * self.step

    def reset_eps(self):
        self.step = 0

    def update(self):
        if self.buffer.length > self.batch_size:
            self.step += 1
            meta_vs, states, actions, rewards, states_next, is_terminals = self.buffer.sample(self.batch_size)
            meta_vs = to_Cuda(torch.as_tensor(meta_vs), self.cuda, self.device)
            states = to_Cuda(torch.as_tensor(states), self.cuda, self.device)
            actions = to_Cuda(torch.as_tensor(actions).long(), self.cuda, self.device)
            states_next = to_Cuda(torch.as_tensor(states_next), self.cuda, self.device)
            rewards = to_Cuda(torch.as_tensor(rewards), self.cuda, self.device)
            is_terminals = to_Cuda(torch.as_tensor(is_terminals), self.cuda, self.device)

            Q_estimate = self.policy(meta_vs, states).gather(1, actions.unsqueeze(1))
            qvalues_target = self.policy_old(meta_vs, states_next)
            Q_target = rewards.unsqueeze(1) + self.gamma * qvalues_target.max(1)[0].unsqueeze(1) * (
                        1 - is_terminals).unsqueeze(1)
            loss = self.MSE_loss(Q_estimate, Q_target.detach()).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.step % self.update_target_freq == 0:
                self.policy_old.load_state_dict(self.policy.state_dict())

    def re_init(self):
        self.policy.init_actor()
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))