import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from utils.utils import init_weights_kaiming_uniform

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights_kaiming_uniform)

    def forward(self):
        raise NotImplementedError

    def get_actor_weights(self):
        w1, w2, w3 = self.actor[0].weight, self.actor[2].weight, self.actor[4].weight
        w1 = torch.flatten(w1).detach().numpy()
        w2 = torch.flatten(w2).detach().numpy()
        w3 = torch.flatten(w3).detach().numpy()
        return [w1, w2, w3]

    def init_actor(self):
        self.actor.apply(init_weights_kaiming_uniform)

    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr_a, lr_c, gamma, K_epochs, eps_clip, device, cuda):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.device = device
        self.cuda = cuda

        self.use_mf = True
        if self.use_mf:
            state_dim += action_dim
        
        self.buffer = RolloutBuffer()
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_a},
            {'params': self.policy.critic.parameters(), 'lr': lr_c}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        if cuda:
            self.policy.to(device)
            self.policy_old.to(device)
            self.MseLoss.to(device)

        # model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # total_num = sum(p.numel() for p in self.policy.parameters())
        # trainable_num = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        # print('Total', total_num, 'Trainable', trainable_num)
        # input('Count Simple')

    def select_action(self, state, former_act_prob, store_tuple=True, store_tuple_idx=0):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            if self.cuda:
                state = state.to(self.device)

            if self.use_mf:
                former_act_prob = torch.FloatTensor(former_act_prob).to(self.device)
                state = torch.cat((state, former_act_prob), dim=1)
                
            action, action_logprob = self.policy_old.act(state)
        if store_tuple:
            self.buffer.states.append(state[store_tuple_idx])
            self.buffer.actions.append(action[store_tuple_idx])
            self.buffer.logprobs.append(action_logprob[store_tuple_idx])
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
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach()
        if self.cuda:
            old_states = old_states.to(self.device)
            old_actions = old_actions.to(self.device)
            old_logprobs = old_logprobs.to(self.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

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