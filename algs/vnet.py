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

    def push(self, n, state, state_next, action, reward, is_terminal):
        for i in range(n):
            self.states[self.flag] = state[i]
            self.states_next[self.flag] = state_next[i]
            self.actions[self.flag] = action[i]
            self.rewards[self.flag] = reward[i]
            self.is_terminals[self.flag] = is_terminal[i]
            self.add_cnt()

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
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim + 2, hidden_dim), # eps and xis
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights)

    def forward(self, state):
        action_values = self.actor(state)
        return action_values

    def init_actor(self):
        self.actor.apply(init_weights)


class Vnet:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, gamma, init_eps, final_eps, eps_decay_step, device, cuda, use_mf, adv_num, agent_num, q_func,
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
        self.adv_num = adv_num
        self.agent_num = agent_num
        self.q_func = q_func
        
        if self.use_mf:
            state_dim += action_dim

        self.buffer = ReplayBuffer(state_dim, action_dim, buffer_size)

        self.policy = ValueNet(state_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.MSE_loss = nn.MSELoss()

        if cuda:
            self.policy.to(device)
            self.MSE_loss.to(device)

    def reset_eps(self):
        self.step = 0

    def update(self):
        if self.buffer.length > self.batch_size:
            states, actions, rewards, states_next, is_terminals = self.buffer.sample(self.batch_size)
            states = to_Cuda(torch.as_tensor(states), self.cuda, self.device)
            actions = to_Cuda(torch.as_tensor(actions).long(), self.cuda, self.device)
            states_next = to_Cuda(torch.as_tensor(states_next), self.cuda, self.device)
            rewards = to_Cuda(torch.as_tensor(rewards), self.cuda, self.device)
            is_terminals = to_Cuda(torch.as_tensor(is_terminals), self.cuda, self.device)

            eps = np.random.choice(range(1, self.adv_num+1), size=len(states), replace=True) / self.agent_num
            xis = []
            for prob in eps:
                sampled_value = np.random.choice([0, 1], p=[1 - prob, prob])
                xis.append(sampled_value)
            xis = np.array(xis)

            last_Q = []
            for i in range(len(states)):
                last_Q.append(self.q_func.Q_value(states[i], actions[i]))

            last_Q = np.array(last_Q)
            last_Q = np.expand_dims(last_Q, axis=1)

            eps = to_Cuda(torch.as_tensor(eps), self.cuda, self.device)
            xis = to_Cuda(torch.as_tensor(xis), self.cuda, self.device)
            last_Q = to_Cuda(torch.as_tensor(last_Q), self.cuda, self.device)

            with torch.no_grad():
                eps = eps.unsqueeze(1)
                xis = xis.unsqueeze(1)
                states = torch.cat((states, eps, xis), dim=1).to(torch.float32)
                v_values = self.policy(states)
                # v_values = v_values.detach().cpu().numpy()
                target_v = rewards.unsqueeze(1) + self.gamma * (1. - is_terminals.unsqueeze(1)) * v_values - (eps * xis + eps + xis) * last_Q

            target_v =  target_v.to(torch.float32)
            v_values = self.policy(states)

            loss = self.MSE_loss(target_v, v_values).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()

    def V(self, state, eps, xis):
        eps = np.full((len(state), 1), eps)
        xis = np.expand_dims(xis, axis = 1)
        states = np.concatenate((state, eps, xis), axis=1)
        states = torch.FloatTensor(states)
        if self.cuda:
            states = states.to(self.device)
        return self.policy(states).detach()
    
    def re_init(self):
        self.policy.init_actor()

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))