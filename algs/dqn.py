import numpy as np
import torch
import torch.nn as nn
from utils.utils import init_weights, to_Cuda



class ReplayBuffer(object):
    def __init__(self, state_dim, act_dim, buffer_size):
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.buffer_size = buffer_size

        self.states = np.zeros((buffer_size,) + (state_dim, )).astype('float32')
        self.states_next = np.zeros((buffer_size,) + (state_dim, )).astype('float32')
        self.actions = np.zeros((buffer_size,) + ()).astype('int32')
        self.rewards = np.zeros((buffer_size,) + ()).astype('float32')
        self.is_terminals = np.zeros((buffer_size,) + ()).astype('float32')
        self.length = 0
        self.flag = 0

    def append_state(self, state):
        self.states[self.flag] = state

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
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        is_terminals = self.is_terminals[idx]
        states_next = self.states_next[idx]
        return states, actions, rewards, states_next, is_terminals


class ValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ValueNet, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(init_weights)

    def forward(self, state):
        action_values = self.actor(state)
        return action_values

    def init_actor(self):
        self.actor.apply(init_weights)


class DQN:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, init_eps, final_eps, eps_decay_step, device, cuda, use_mf,
                 buffer_size=int(1e6), batch_size=64, update_target_freq=100):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
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

        self.use_mf = use_mf
        
        if self.use_mf:
            state_dim += action_dim

        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

        self.policy, self.policy_old = ValueNet(state_dim, action_dim, hidden_dim), ValueNet(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MSE_loss = nn.MSELoss()

        if cuda:
            self.policy.to(device)
            self.policy_old.to(device)
            self.MSE_loss.to(device)

        # model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # print('MLP Param Num: ', params)
        # input('Count MLP')

    def select_action(self, states, former_act_prob=None, store_tuple=True, store_tuple_idx=0):
        if self.use_mf:
            states = np.concatenate((states, former_act_prob), axis=1)

        if store_tuple:
            self.buffer.append_state(states[store_tuple_idx])
        n = len(states)
        states = torch.FloatTensor(states)
        if self.cuda:
            states = states.to(self.device)
        
        action_values = self.policy(states).detach()
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
            states, actions, rewards, states_next, is_terminals = self.buffer.sample(self.batch_size)
            states = to_Cuda(torch.as_tensor(states), self.cuda, self.device)
            actions = to_Cuda(torch.as_tensor(actions).long(), self.cuda, self.device)
            states_next = to_Cuda(torch.as_tensor(states_next), self.cuda, self.device)
            rewards = to_Cuda(torch.as_tensor(rewards), self.cuda, self.device)
            is_terminals = to_Cuda(torch.as_tensor(is_terminals), self.cuda, self.device)

            Q_estimate = self.policy(states).gather(1, actions.unsqueeze(1))
            qvalues_target = self.policy_old(states_next)
            Q_target = rewards.unsqueeze(1) + self.gamma * qvalues_target.max(1)[0].unsqueeze(1) * (1 - is_terminals).unsqueeze(1)
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