import torch
import torch.nn as nn
from torch.distributions import Categorical
from .hyper_nets import HyperNetwork


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


class HyperAC(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim=256):
        super(HyperAC, self).__init__()

        self.actor = HyperNetwork(meta_v_dim, state_dim, action_dim, hidden_dim)
        self.critic = HyperNetwork(meta_v_dim, state_dim, 1, hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state):
        action_probs, _ = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action):
        action_probs, actor_layer_out = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values, critic_layer_out = self.critic(meta_v, state)

        return action_logprobs, state_values, dist_entropy


class HyperPPO:
    def __init__(self, meta_v_dim, state_dim, action_dim, lr_a, lr_c, gamma, K_epochs, eps_clip, device, cuda, hidden_dim=256, w_decay=0):
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

        self.buffer = RolloutBuffer()

        self.policy = HyperAC(meta_v_dim, state_dim, action_dim, hidden_dim)
        self.policy_old = HyperAC(meta_v_dim, state_dim, action_dim, hidden_dim)

        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_a},
            {'params': self.policy.critic.parameters(), 'lr': lr_c}
        ], weight_decay=w_decay)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.KLDivLoss = nn.KLDivLoss(reduction='batchmean')  # note that use batchmean

        if cuda:
            self.policy.to(device)
            self.policy_old.to(device)
            self.MseLoss.to(device)
            self.KLDivLoss.to(device)

    def select_action(self, meta_v, state):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            if self.cuda:
                state = state.to(self.device)
            action, action_logprob = self.policy_old.act(meta_v, state)
        self.buffer.meta_vs.append(meta_v[0])
        self.buffer.states.append(state[0])
        self.buffer.actions.append(action[0])
        self.buffer.logprobs.append(action_logprob[0])

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
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_meta_vs, old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs)

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = (-torch.min(surr1, surr2) - 0.01 * dist_entropy).mean() + 0.5 * self.MseLoss(state_values, rewards)

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))