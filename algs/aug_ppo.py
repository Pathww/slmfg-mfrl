import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils.utils import init_weights_kaiming_uniform

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


#=============================
class ActorAugEmb(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim):
        super(ActorAugEmb, self).__init__()
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.actor = nn.Sequential(
            nn.Linear(state_dim + meta_v_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(init_weights_kaiming_uniform)

    def forward(self, meta_v, state):
        meta_v_emb = self.meta_v_emb_layer(meta_v)
        logits = self.actor(torch.cat([meta_v_emb, state], dim=1))
        return logits

    def get_layer_out(self, meta_v, state):
        with torch.no_grad():
            meta_v_emb = self.meta_v_emb_layer(meta_v)
            lay_out_list = []
            x = torch.cat([meta_v_emb, state], dim=1)
            for layer in self.actor:
                x = layer(x)
                lay_out_list.append(x)
        return [lay_out_list[1], lay_out_list[3], lay_out_list[4]]

    def get_weights(self):
        w1, b1 = self.actor[0].weight, self.actor[0].bias
        w2, b2 = self.actor[2].weight, self.actor[2].bias
        w3, b3 = self.actor[4].weight, self.actor[4].bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class CriticAugEmb(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, hidden_dim):
        super(CriticAugEmb, self).__init__()
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + meta_v_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights_kaiming_uniform)

    def forward(self, meta_v, state):
        meta_v_emb = self.meta_v_emb_layer(meta_v)
        values = self.critic(torch.cat([meta_v_emb, state], dim=1))
        return values

    def get_weights(self):
        w1, b1 = self.critic[0].weight, self.critic[0].bias
        w2, b2 = self.critic[2].weight, self.critic[2].bias
        w3, b3 = self.critic[4].weight, self.critic[4].bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class ActorCriticAugACEmb(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim):
        super(ActorCriticAugACEmb, self).__init__()

        self.actor = ActorAugEmb(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim)
        self.critic = CriticAugEmb(meta_v_dim, meta_v_emb_dim, state_dim, hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        self.actor.apply(init_weights_kaiming_uniform)

    def act(self, meta_v, state):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        # print(dist.logits[0], '  ', dist.probs[0])
        # input('feererere')
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy


class ActorAugEmbCH(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim):
        super(ActorAugEmbCH, self).__init__()
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.fc1 = nn.Linear(state_dim + meta_v_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + meta_v_emb_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + meta_v_emb_dim, action_dim)
        self.apply(init_weights_kaiming_uniform)

    def forward(self, meta_v, state):
        meta_v_emb = self.meta_v_emb_layer(meta_v)
        out = F.relu(self.fc1(torch.cat([meta_v_emb, state], dim=1)))
        out = F.relu(self.fc2(torch.cat([meta_v_emb.clone().detach(), out], dim=1)))
        logits = self.fc3(torch.cat([meta_v_emb.clone().detach(), out], dim=1))
        return logits

    def get_layer_out(self, meta_v, state):
        with torch.no_grad():
            meta_v_emb = self.meta_v_emb_layer(meta_v)
            out_0 = self.fc1(torch.cat([meta_v_emb, state], dim=1))
            out_1 = F.relu(out_0)
            out_1_concat = torch.cat([meta_v_emb, out_1], dim=1)
            out_2 = self.fc2(out_1_concat)
            out_3 = F.relu(out_2)
            out_3_concat = torch.cat([meta_v_emb, out_3], dim=1)
            out_4 = self.fc3(out_3_concat)
        return [out_1, out_3, out_4]

    def get_weights(self):
        w1, b1 = self.fc1.weight, self.fc1.bias
        w2, b2 = self.fc2.weight, self.fc2.bias
        w3, b3 = self.fc3.weight, self.fc3.bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class CriticAugEmbCH(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, hidden_dim):
        super(CriticAugEmbCH, self).__init__()
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.fc1 = nn.Linear(state_dim + meta_v_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + meta_v_emb_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + meta_v_emb_dim, 1)
        self.apply(init_weights_kaiming_uniform)

    def forward(self, meta_v, state):
        meta_v_emb = self.meta_v_emb_layer(meta_v)
        out = F.relu(self.fc1(torch.cat([meta_v_emb, state], dim=1)))
        out = F.relu(self.fc2(torch.cat([meta_v_emb.clone().detach(), out], dim=1)))
        values = self.fc3(torch.cat([meta_v_emb.clone().detach(), out], dim=1))
        return values

    def get_weights(self):
        w1, b1 = self.fc1.weight, self.fc1.bias
        w2, b2 = self.fc2.weight, self.fc2.bias
        w3, b3 = self.fc3.weight, self.fc3.bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class ActorCriticAugACEmbCH(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim):
        super(ActorCriticAugACEmbCH, self).__init__()

        self.actor = ActorAugEmbCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim)
        self.critic = CriticAugEmbCH(meta_v_dim, meta_v_emb_dim, state_dim, hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        self.actor.apply(init_weights_kaiming_uniform)

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
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy


# =============================
class ActorAug(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim):
        super(ActorAug, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim + meta_v_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.apply(init_weights_kaiming_uniform)

    def forward(self, meta_v, state):
        aug_state = torch.cat([meta_v, state], dim=1)
        logits = self.actor(aug_state)
        return logits

    def get_layer_out(self, meta_v, state):
        with torch.no_grad():
            aug_state = torch.cat([meta_v, state], dim=1)
            lay_out_list = []
            x = aug_state
            for layer in self.actor:
                x = layer(x)
                lay_out_list.append(x)
        return [lay_out_list[1], lay_out_list[3], lay_out_list[4]]

    def get_weights(self):
        w1, b1 = self.actor[0].weight, self.actor[0].bias
        w2, b2 = self.actor[2].weight, self.actor[2].bias
        w3, b3 = self.actor[4].weight, self.actor[4].bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class CriticAug(nn.Module):
    def __init__(self, meta_v_dim, state_dim, hidden_dim):
        super(CriticAug, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_dim + meta_v_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.apply(init_weights_kaiming_uniform)

    def forward(self, meta_v, state):
        aug_state = torch.cat([meta_v, state], dim=1)
        values = self.critic(aug_state)
        return values

    def get_weights(self):
        w1, b1 = self.critic[0].weight, self.critic[0].bias
        w2, b2 = self.critic[2].weight, self.critic[2].bias
        w3, b3 = self.critic[4].weight, self.critic[4].bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class ActorCriticAugAC(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim):
        super(ActorCriticAugAC, self).__init__()

        self.actor = ActorAug(meta_v_dim, state_dim, action_dim, hidden_dim)
        self.critic = CriticAug(meta_v_dim, state_dim, hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        self.actor.apply(init_weights_kaiming_uniform)

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
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy


class ActorAugCH(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim):
        super(ActorAugCH, self).__init__()
        self.fc1 = nn.Linear(state_dim + meta_v_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + meta_v_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + meta_v_dim, action_dim)
        self.apply(init_weights_kaiming_uniform)

    def forward(self, meta_v, state):
        out = F.relu(self.fc1(torch.cat([meta_v, state], dim=1)))
        out = F.relu(self.fc2(torch.cat([meta_v, out], dim=1)))
        logits = self.fc3(torch.cat([meta_v, out], dim=1))
        return logits

    def get_layer_out(self, meta_v, state):
        with torch.no_grad():
            aug_state = torch.cat([meta_v, state], dim=1)
            out_0 = self.fc1(aug_state)
            out_1 = F.relu(out_0)
            out_1_concat = torch.cat([meta_v, out_1], dim=1)
            out_2 = self.fc2(out_1_concat)
            out_3 = F.relu(out_2)
            out_3_concat = torch.cat([meta_v, out_3], dim=1)
            out_4 = self.fc3(out_3_concat)
        return [out_1, out_3, out_4]

    def get_weights(self):
        w1, b1 = self.fc1.weight, self.fc1.bias
        w2, b2 = self.fc2.weight, self.fc2.bias
        w3, b3 = self.fc3.weight, self.fc3.bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class CriticAugCH(nn.Module):
    def __init__(self, meta_v_dim, state_dim, hidden_dim):
        super(CriticAugCH, self).__init__()
        self.fc1 = nn.Linear(state_dim + meta_v_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim + meta_v_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim + meta_v_dim, 1)
        self.apply(init_weights_kaiming_uniform)

    def forward(self, meta_v, state):
        out = F.relu(self.fc1(torch.cat([meta_v, state], dim=1)))
        out = F.relu(self.fc2(torch.cat([meta_v, out], dim=1)))
        values = self.fc3(torch.cat([meta_v, out], dim=1))
        return values

    def get_weights(self):
        w1, b1 = self.fc1.weight, self.fc1.bias
        w2, b2 = self.fc2.weight, self.fc2.bias
        w3, b3 = self.fc3.weight, self.fc3.bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class ActorCriticAugACCH(nn.Module):
    def __init__(self, meta_v_dim, state_dim, action_dim, hidden_dim):
        super(ActorCriticAugACCH, self).__init__()

        self.actor = ActorAugCH(meta_v_dim, state_dim, action_dim, hidden_dim)
        self.critic = CriticAugCH(meta_v_dim, state_dim, hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        self.actor.apply(init_weights_kaiming_uniform)

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
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy


class AugPPO:
    def __init__(self, meta_v_dim, meta_v_emb, meta_v_emb_dim, state_dim, action_dim, hidden_dim, lr_a, lr_c, gamma,
                 K_epochs, eps_clip, device, cuda, min_meta_v, max_meta_v, pos_emb, pos_emb_dim, concat_meta_v_in_hidden=False):
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb = meta_v_emb
        self.meta_v_emb_dim = meta_v_emb_dim
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
        self.min_meta_v = min_meta_v
        self.max_meta_v = max_meta_v
        self.pos_emb = pos_emb
        self.pos_emb_dim = pos_emb_dim
        self.concat_meta_v_in_hidden = concat_meta_v_in_hidden

        self.buffer = RolloutBuffer()

        meta_input_dim = meta_v_dim if not pos_emb else pos_emb_dim

        if meta_v_emb:
            if not concat_meta_v_in_hidden:
                self.policy = ActorCriticAugACEmb(meta_input_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim)
                self.policy_old = ActorCriticAugACEmb(meta_input_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim)
            else:
                self.policy = ActorCriticAugACEmbCH(meta_input_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim)
                self.policy_old = ActorCriticAugACEmbCH(meta_input_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim)
        else:
            if not concat_meta_v_in_hidden:
                self.policy = ActorCriticAugAC(meta_input_dim, state_dim, action_dim, hidden_dim)
                self.policy_old = ActorCriticAugAC(meta_input_dim, state_dim, action_dim, hidden_dim)
            else:
                self.policy = ActorCriticAugACCH(meta_input_dim, state_dim, action_dim, hidden_dim)
                self.policy_old = ActorCriticAugACCH(meta_input_dim, state_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_a},
            {'params': self.policy.critic.parameters(), 'lr': lr_c}
        ])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.MseLoss = nn.MSELoss()
        if cuda:
            self.policy.to(device)
            self.policy_old.to(device)
            self.MseLoss.to(device)

        self.N_emb = np.array([[int(x) for x in '{0:012b}'.format(i)] for i in range(4096)])

        # model_parameters = filter(lambda p: p.requires_grad, self.policy.parameters())
        # params = sum([np.prod(p.size()) for p in model_parameters])
        # total_num = sum(p.numel() for p in self.policy.parameters())
        # trainable_num = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        # print('Total', total_num, 'Trainable', trainable_num)
        # input('Count Simple')

    def get_policy_weights(self):
        actor_w, actor_b = self.policy.actor.get_weights()
        critic_w, critic_b = self.policy.critic.get_weights()
        return actor_w, actor_b, critic_w, critic_b

    def select_action(self, meta_v, state, store_tuple=True, store_tuple_idx=0, ret_prob=False, pos_emb=None):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            if self.cuda:
                state = state.to(self.device)
            action, action_logprob = self.policy_old.act(meta_v, state)
        if store_tuple:
            self.buffer.meta_vs.append(meta_v[store_tuple_idx])
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

        # Normalizing the returns
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
        # if not self.pos_emb:
        if self.meta_v_dim == 1:
            old_meta_vs = old_meta_vs.unsqueeze(1)

        # print(old_meta_vs.shape)
        # print(old_states.shape)
        # print(old_actions.shape)
        # print(old_logprobs.shape)
        # input('dddddd')

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_meta_vs, old_states, old_actions)

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