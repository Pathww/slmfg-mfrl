import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from .simple_hyper_nets import HyperNetwork1Layer, HyperNetwork1LayerCH, HyperNetworkEmbedInput1Layer, HyperNetworkEmbedInput1LayerCH
from .simple_hyper_nets import HyperNetwork2Layer, HyperNetwork2LayerCH, HyperNetworkEmbedInput2Layer, HyperNetworkEmbedInput2LayerCH
from .simple_hyper_nets import HyperNetwork, HyperNetworkCH, HyperNetworkEmbedInput, HyperNetworkEmbedInputCH
from .simple_hyper_nets import HyperNetwork1LayerNoAug, HyperNetworkEmbedInput1LayerNoAug
from .simple_hyper_nets import HyperNetwork2LayerNoAug, HyperNetworkEmbedInput2LayerNoAug
from .simple_hyper_nets import HyperNetworkNoAug, HyperNetworkEmbedInputNoAug
from utils.utils import l1_reg_loss, get_1d_sincos_pos_embed_from_grid


class RolloutBuffer:
    def __init__(self):
        self.meta_vs = []
        self.pos_emb = []
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.meta_vs[:]
        del self.pos_emb[:]
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


#=============== 1 Layers =====================
class HyperAC1Layer(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, state_dim, action_dim, z_dim=128, dynamic_hidden_dim=128):
        super(HyperAC1Layer, self).__init__()
        self.actor = HyperNetwork1Layer(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetwork1Layer(meta_v_dim, state_dim, 1, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad)


class HyperACEmbedInput1Layer(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim=128, dynamic_hidden_dim=128, clip_emb=False, scale=True, bp_emb='hyper'):
        super(HyperACEmbedInput1Layer, self).__init__()
        self.scale = scale
        self.bp_emb = bp_emb
        self.actor = HyperNetworkEmbedInput1Layer(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInput1Layer(meta_v_dim, meta_v_emb_dim, state_dim, 1, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


class HyperAC1LayerCH(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, state_dim, action_dim, z_dim=128, dynamic_hidden_dim=128):
        super(HyperAC1LayerCH, self).__init__()
        self.actor = HyperNetwork1LayerCH(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetwork1LayerCH(meta_v_dim, state_dim, 1, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, actor_get_layer_out=False, actor_layer=None, critic_get_layer_out=False, critic_layer=None, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad)


class HyperACEmbedInput1LayerCH(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim=128, dynamic_hidden_dim=128, clip_emb=False, scale=True, bp_emb='hyper'):
        super(HyperACEmbedInput1LayerCH, self).__init__()
        self.scale = scale
        self.bp_emb = bp_emb
        self.actor = HyperNetworkEmbedInput1LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInput1LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, 1, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


class HyperAC1LayerNoAug(nn.Module):
    # meta_v --> hypernet
    def __init__(self, meta_v_dim, state_dim, action_dim, z_dim=128, dynamic_hidden_dim=128):
        super(HyperAC1LayerNoAug, self).__init__()
        self.actor = HyperNetwork1LayerNoAug(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetwork1LayerNoAug(meta_v_dim, state_dim, 1, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad)


class HyperACEmbedInput1LayerNoAug(nn.Module):
    # meta_v --> embedding --> hypernet
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim=128, dynamic_hidden_dim=128, clip_emb=False, scale=True):
        super(HyperACEmbedInput1LayerNoAug, self).__init__()
        self.scale = scale
        self.actor = HyperNetworkEmbedInput1LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInput1LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, 1, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


#=============== 2 Layers =====================
class HyperAC2Layer(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, state_dim, action_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128):
        super(HyperAC2Layer, self).__init__()
        self.actor = HyperNetwork2Layer(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetwork2Layer(meta_v_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad)


class HyperACEmbedInput2Layer(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128, clip_emb=False, scale=True, bp_emb='hyper'):
        super(HyperACEmbedInput2Layer, self).__init__()
        self.scale = scale
        self.bp_emb = bp_emb
        self.actor = HyperNetworkEmbedInput2Layer(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInput2Layer(meta_v_dim, meta_v_emb_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        # print('papo act')
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        # print(dist.logits[0], '  ', dist.probs[0])
        # input('feererere')
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # input('critic')
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        # input('FFFFF')
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


class HyperAC2LayerCH(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, state_dim, action_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128):
        super(HyperAC2LayerCH, self).__init__()
        self.actor = HyperNetwork2LayerCH(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetwork2LayerCH(meta_v_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad)


class HyperACEmbedInput2LayerCH(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128, clip_emb=False, scale=True, bp_emb='hyper', relu_emb_in_hidden=False):
        super(HyperACEmbedInput2LayerCH, self).__init__()
        self.scale = scale
        self.bp_emb = bp_emb
        self.relu_emb_in_hidden = relu_emb_in_hidden
        self.actor = HyperNetworkEmbedInput2LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInput2LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb, relu_emb_in_hidden=self.relu_emb_in_hidden)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb, relu_emb_in_hidden=self.relu_emb_in_hidden)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb, relu_emb_in_hidden=self.relu_emb_in_hidden)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


class HyperAC2LayerNoAug(nn.Module):
    # meta_v --> hypernet
    def __init__(self, meta_v_dim, state_dim, action_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128):
        super(HyperAC2LayerNoAug, self).__init__()
        self.actor = HyperNetwork2LayerNoAug(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetwork2LayerNoAug(meta_v_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad)


class HyperACEmbedInput2LayerNoAug(nn.Module):
    # meta_v --> embedding --> hypernet
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128, clip_emb=False, scale=True):
        super(HyperACEmbedInput2LayerNoAug, self).__init__()
        self.scale = scale
        self.actor = HyperNetworkEmbedInput2LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInput2LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


#=============== 4 Layers =====================
class HyperAC4Layer(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, state_dim, action_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256):
        super(HyperAC4Layer, self).__init__()
        self.actor = HyperNetwork(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetwork(meta_v_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad)


class HyperACEmbedInput4Layer(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256, clip_emb=False, scale=True, bp_emb='hyper'):
        super(HyperACEmbedInput4Layer, self).__init__()
        self.scale = scale
        self.bp_emb = bp_emb
        self.actor = HyperNetworkEmbedInput(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInput(meta_v_dim, meta_v_emb_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


class HyperAC4LayerCH(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, state_dim, action_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256):
        super(HyperAC4LayerCH, self).__init__()
        self.actor = HyperNetworkCH(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetworkCH(meta_v_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad)


class HyperACEmbedInput4LayerCH(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256, clip_emb=False, scale=True, bp_emb='hyper'):
        super(HyperACEmbedInput4LayerCH, self).__init__()
        self.scale = scale
        self.bp_emb = bp_emb
        self.actor = HyperNetworkEmbedInputCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInputCH(meta_v_dim, meta_v_emb_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale, bp_emb=self.bp_emb)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


class HyperAC4LayerNoAug(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, state_dim, action_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256):
        super(HyperAC4LayerNoAug, self).__init__()
        self.actor = HyperNetworkNoAug(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
        self.critic = HyperNetworkNoAug(meta_v_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad)


class HyperACEmbedInput4LayerNoAug(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256, clip_emb=False, scale=True):
        super(HyperACEmbedInput4LayerNoAug, self).__init__()
        self.scale = scale
        self.actor = HyperNetworkEmbedInputNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)
        self.critic = HyperNetworkEmbedInputNoAug(meta_v_dim, meta_v_emb_dim, state_dim, 1, hyper_hidden_dim, z_dim, dynamic_hidden_dim, clip_emb=clip_emb)

    def forward(self):
        raise NotImplementedError

    def init_actor(self):
        pass

    def act(self, meta_v, state, ret_prob=False, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        dist = Categorical(logits=action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        if ret_prob:
            probs = F.softmax(action_probs, dim=1)
            return action.detach(), action_logprob.detach(), probs.detach()
        return action.detach(), action_logprob.detach()

    def evaluate(self, meta_v, state, action, pos_emb=None):
        action_probs = self.actor(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        dist = Categorical(logits=action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(meta_v, state, pos_emb=pos_emb, scale=self.scale)
        return action_logprobs, state_values, dist_entropy

    def get_dynamic_weights(self, meta_v, no_grad=True, pos_emb=None):
        return self.actor.get_weight(meta_v, no_grad=no_grad, pos_emb=pos_emb, scale=self.scale)


#=======
class Actor(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim, cuda, device):
        super(Actor, self).__init__()
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb_dim = meta_v_emb_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        if cuda:
            self.meta_v_emb_layer.to(device)
        self.input_layer_w = torch.nn.Parameter(torch.ones((1, self.hidden_dim, self.meta_v_emb_dim + self.state_dim)).to(device))
        self.input_layer_b = torch.nn.Parameter(torch.ones((1, self.hidden_dim, 1)).to(device))
        self.input_layer_s = torch.nn.Parameter(torch.ones((1, self.hidden_dim, 1)).to(device))
        self.hidden_layer_w = torch.nn.Parameter(torch.ones((1, self.hidden_dim, self.hidden_dim)).to(device))
        self.hidden_layer_b = torch.nn.Parameter(torch.ones((1, self.hidden_dim, 1)).to(device))
        self.hidden_layer_s = torch.nn.Parameter(torch.ones((1, self.hidden_dim, 1)).to(device))
        self.output_layer_w = torch.nn.Parameter(torch.ones((1, self.action_dim, self.hidden_dim)).to(device))
        self.output_layer_b = torch.nn.Parameter(torch.ones((1, self.action_dim, 1)).to(device))
        self.output_layer_s = torch.nn.Parameter(torch.ones((1, self.action_dim, 1)).to(device))

    def init_emb_layer(self, emb_layer):
        self.meta_v_emb_layer.load_state_dict(emb_layer.state_dict())

    def init_network(self, w, b, s):
        self.input_layer_w.data.copy_(w[0].data)
        self.input_layer_b.data.copy_(b[0].data)
        self.input_layer_s.data.copy_(s[0].data)
        self.hidden_layer_w.data.copy_(w[1].data)
        self.hidden_layer_b.data.copy_(b[1].data)
        self.hidden_layer_s.data.copy_(s[1].data)
        self.output_layer_w.data.copy_(w[2].data)
        self.output_layer_b.data.copy_(b[2].data)
        self.output_layer_s.data.copy_(s[2].data)

    def forward(self, meta_v, state):
        meta_v_emb = self.meta_v_emb_layer(meta_v)
        base_v = torch.cat([state, meta_v_emb.detach()], dim=1)
        out = F.relu(torch.matmul(self.input_layer_w, base_v.unsqueeze(2)) * (1.0 + self.input_layer_s) + self.input_layer_b)
        out = F.relu(torch.matmul(self.hidden_layer_w, out) * (1.0 + self.hidden_layer_s) + self.hidden_layer_b)
        out = torch.matmul(self.output_layer_w, out) * (1.0 + self.output_layer_s) + self.output_layer_b
        logits = torch.squeeze(out, dim=2)
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


class Critic(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, hidden_dim, cuda, device):
        super(Critic, self).__init__()
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb_dim = meta_v_emb_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.device = device

        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        if cuda:
            self.meta_v_emb_layer.to(device)
        self.input_layer_w = torch.nn.Parameter(torch.ones((1, self.hidden_dim, self.meta_v_emb_dim + self.state_dim)).to(device))
        self.input_layer_b = torch.nn.Parameter(torch.ones((1, self.hidden_dim, 1)).to(device))
        self.input_layer_s = torch.nn.Parameter(torch.ones((1, self.hidden_dim, 1)).to(device))
        self.hidden_layer_w = torch.nn.Parameter(torch.ones((1, self.hidden_dim, self.hidden_dim)).to(device))
        self.hidden_layer_b = torch.nn.Parameter(torch.ones((1, self.hidden_dim, 1)).to(device))
        self.hidden_layer_s = torch.nn.Parameter(torch.ones((1, self.hidden_dim, 1)).to(device))
        self.output_layer_w = torch.nn.Parameter(torch.ones((1, 1, self.hidden_dim)).to(device))
        self.output_layer_b = torch.nn.Parameter(torch.ones((1, 1, 1)).to(device))
        self.output_layer_s = torch.nn.Parameter(torch.ones((1, 1, 1)).to(device))

    def init_emb_layer(self, emb_layer):
        self.meta_v_emb_layer.load_state_dict(emb_layer.state_dict())

    def init_network(self, w, b, s):
        self.input_layer_w.data.copy_(w[0].data)
        self.input_layer_b.data.copy_(b[0].data)
        self.input_layer_s.data.copy_(s[0].data)
        self.hidden_layer_w.data.copy_(w[1].data)
        self.hidden_layer_b.data.copy_(b[1].data)
        self.hidden_layer_s.data.copy_(s[1].data)
        self.output_layer_w.data.copy_(w[2].data)
        self.output_layer_b.data.copy_(b[2].data)
        self.output_layer_s.data.copy_(s[2].data)

    def forward(self, meta_v, state):
        meta_v_emb = self.meta_v_emb_layer(meta_v)
        base_v = torch.cat([state, meta_v_emb.detach()], dim=1)
        out = F.relu(torch.matmul(self.input_layer_w, base_v.unsqueeze(2)) * (1.0 + self.input_layer_s) + self.input_layer_b)
        out = F.relu(torch.matmul(self.hidden_layer_w, out) * (1.0 + self.hidden_layer_s) + self.hidden_layer_b)
        out = torch.matmul(self.output_layer_w, out) * (1.0 + self.output_layer_s) + self.output_layer_b
        values = torch.squeeze(out, dim=2)
        return values

    def get_weights(self):
        w1, b1 = self.critic[0].weight, self.critic[0].bias
        w2, b2 = self.critic[2].weight, self.critic[2].bias
        w3, b3 = self.critic[4].weight, self.critic[4].bias
        return [torch.flatten(w1).detach(), torch.flatten(w2).detach(), torch.flatten(w3).detach()], \
               [torch.flatten(b1).detach(), torch.flatten(b2).detach(), torch.flatten(b3).detach()]


class ActorCritic(nn.Module):
    def __init__(self, meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim, cuda, device):
        super(ActorCritic, self).__init__()

        self.actor = Actor(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hidden_dim, cuda, device)
        self.critic = Critic(meta_v_dim, meta_v_emb_dim, state_dim, hidden_dim, cuda, device)

    def forward(self):
        raise NotImplementedError

    def act(self, meta_v, state):
        # print('papo ft act')
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
#======


class SimpleHyperPPO2Layer:
    def __init__(self, meta_v_dim, meta_v_emb, meta_v_emb_dim, state_dim, action_dim, lr_a, lr_c, lr_a_finetune, lr_c_finetune, gamma, K_epochs, eps_clip, device, cuda,
                 min_meta_v, max_meta_v, pos_emb, pos_emb_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256, hyper_actor=True,
                 hyper_critic=True, reg_actor=False, reg_actor_layer=None, reg_critic=False, reg_critic_layer=None, fix_critic=False,
                 optimizer='adam', w_decay=0, l1=False, w_clip=False, w_clip_p=10, max_pe_encode=10000, concat_meta_v_in_hidden=False,
                 clip_emb=False, num_hidden=4, scale=True, bp_emb='hyper', aug=True, relu_emb_in_hidden=False):
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb = meta_v_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.lr_a_finetune = lr_a_finetune
        self.lr_c_finetune = lr_c_finetune
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
        self.concat_meta_v_in_hidden = concat_meta_v_in_hidden
        self.w_decay = w_decay

        if pos_emb:
            assert meta_v_emb and meta_v_emb_dim == pos_emb_dim, "Positional encoding must be used with meta_v_emb."

        self.buffer = RolloutBuffer()

        if not meta_v_emb:
            if concat_meta_v_in_hidden:
                if num_hidden == 1:
                    self.policy = HyperAC1LayerCH(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
                    self.policy_old = HyperAC1LayerCH(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
                elif num_hidden == 2:
                    self.policy = HyperAC2LayerCH(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
                    self.policy_old = HyperAC2LayerCH(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
                elif num_hidden == 4:
                    self.policy = HyperAC4LayerCH(meta_v_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim)
                    self.policy_old = HyperAC4LayerCH(meta_v_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim)
            else:
                if num_hidden == 1:
                    if aug:
                        self.policy = HyperAC1Layer(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
                        self.policy_old = HyperAC1Layer(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
                    else:
                        self.policy = HyperAC1LayerNoAug(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
                        self.policy_old = HyperAC1LayerNoAug(meta_v_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim)
                elif num_hidden == 2:
                    if aug:
                        self.policy = HyperAC2Layer(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
                        self.policy_old = HyperAC2Layer(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
                    else:
                        self.policy = HyperAC2LayerNoAug(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
                        self.policy_old = HyperAC2LayerNoAug(meta_v_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim)
                elif num_hidden == 4:
                    if aug:
                        self.policy = HyperAC4Layer(meta_v_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim)
                        self.policy_old = HyperAC4Layer(meta_v_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim)
                    else:
                        self.policy = HyperAC4LayerNoAug(meta_v_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim)
                        self.policy_old = HyperAC4LayerNoAug(meta_v_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim)
        else:
            if concat_meta_v_in_hidden:
                if num_hidden == 1:
                    self.policy = HyperACEmbedInput1LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                    self.policy_old = HyperACEmbedInput1LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                elif num_hidden == 2:
                    self.policy = HyperACEmbedInput2LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, scale=scale, bp_emb=bp_emb, relu_emb_in_hidden=relu_emb_in_hidden)
                    self.policy_old = HyperACEmbedInput2LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, scale=scale, bp_emb=bp_emb, relu_emb_in_hidden=relu_emb_in_hidden)
                elif num_hidden == 4:
                    self.policy = HyperACEmbedInput4LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                    self.policy_old = HyperACEmbedInput4LayerCH(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
            else:
                if num_hidden == 1:
                    if aug:
                        self.policy = HyperACEmbedInput1Layer(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                        self.policy_old = HyperACEmbedInput1Layer(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                    else:
                        self.policy = HyperACEmbedInput1LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, scale=scale)
                        self.policy_old = HyperACEmbedInput1LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, z_dim, dynamic_hidden_dim, scale=scale)
                elif num_hidden == 2:
                    if aug:
                        self.policy = HyperACEmbedInput2Layer(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                        self.policy_old = HyperACEmbedInput2Layer(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                    else:
                        self.policy = HyperACEmbedInput2LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, scale=scale)
                        self.policy_old = HyperACEmbedInput2LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, hyper_hidden_dim, z_dim, dynamic_hidden_dim, scale=scale)
                elif num_hidden == 4:
                    if aug:
                        self.policy = HyperACEmbedInput4Layer(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                        self.policy_old = HyperACEmbedInput4Layer(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim, scale=scale, bp_emb=bp_emb)
                    else:
                        self.policy = HyperACEmbedInput4LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim, scale=scale)
                        self.policy_old = HyperACEmbedInput4LayerNoAug(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, dynamic_hidden_dim=dynamic_hidden_dim, scale=scale)

        # === for fine-tuning ===
        self.policy_finetune = ActorCritic(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, dynamic_hidden_dim, cuda, device)
        self.policy_finetune_old = ActorCritic(meta_v_dim, meta_v_emb_dim, state_dim, action_dim, dynamic_hidden_dim, cuda, device)
        # print(self.policy_finetune.actor.parameters())
        # for name, param in self.policy_finetune.actor.named_parameters():
        #     print(name)
        # input('fsdfafafefwffsa')
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
        # total_num = sum(p.numel() for p in self.policy.parameters())
        # trainable_num = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        # print('Total', total_num, 'Trainable', trainable_num)
        # input('Count Simple')

    def init_policy_finetune(self, meta_v):
        actor_w, actor_b, actor_s = self.policy.actor.get_weight(meta_v, keep_shape=True)
        critic_w, critic_b, critic_s = self.policy.critic.get_weight(meta_v, keep_shape=True)
        self.policy_finetune.actor.init_emb_layer(self.policy.actor.meta_v_emb_layer)
        self.policy_finetune.actor.init_network(actor_w, actor_b, actor_s)
        self.policy_finetune.critic.init_emb_layer(self.policy.critic.meta_v_emb_layer)
        self.policy_finetune.critic.init_network(critic_w, critic_b, critic_s)
        self.policy_finetune_old.actor.init_emb_layer(self.policy.actor.meta_v_emb_layer)
        self.policy_finetune_old.actor.init_network(actor_w, actor_b, actor_s)
        self.policy_finetune_old.critic.init_emb_layer(self.policy.critic.meta_v_emb_layer)
        self.policy_finetune_old.critic.init_network(critic_w, critic_b, critic_s)
        self.optimizer_finetune = torch.optim.Adam([
            {'params': self.policy_finetune.actor.parameters(), 'lr': self.lr_a_finetune},
            {'params': self.policy_finetune.critic.parameters(), 'lr': self.lr_c_finetune}
        ], weight_decay=self.w_decay)

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
            # if self.pos_emb:
            #     rand_N = np.random.randint(self.min_meta_v, self.max_meta_v + 1, pretrain_batch_size)
            #     rand_N_emb = self.N_emb_sincos[rand_N, :]
            #     meta_v_tensor = torch.FloatTensor(rand_N_emb)
            # else:
            rand_N = np.random.randint(self.min_meta_v, self.max_meta_v + 1, pretrain_batch_size)
            if self.pos_emb:
                pos_emb_tensor = torch.FloatTensor(self.N_emb_sincos[rand_N, :])
                if self.cuda:
                    pos_emb_tensor = pos_emb_tensor.to(self.device)
            else:
                pos_emb_tensor = None
            if self.meta_v_dim > 1:
                # rand_N = np.random.randint(self.min_meta_v, self.max_meta_v + 1, pretrain_batch_size)
                rand_N_emb = self.N_emb[rand_N, :]
                meta_v_tensor = torch.FloatTensor(rand_N_emb)
            else:
                # meta_v_tensor = torch.randint(self.min_meta_v, self.max_meta_v + 1, [pretrain_batch_size, 1], dtype=torch.float32)
                meta_v_tensor = torch.FloatTensor(rand_N).unsqueeze(1)
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
            if loss_type == 'kl':
                with torch.no_grad():
                    label_a = F.softmax(ppo.policy.actor(samples), dim=1)
                    label_c = F.softmax(ppo.policy.critic(samples), dim=1)
                if not self.pos_emb:
                    if not self.meta_v_emb:
                        out_a = torch.log_softmax(self.policy.actor(meta_v_tensor, samples), dim=1)
                        out_c = torch.log_softmax(self.policy.critic(meta_v_tensor, samples), dim=1)
                    else:
                        out_a = torch.log_softmax(self.policy.actor(meta_v_tensor, samples, stop_meta_v_emb_grad=True), dim=1)
                        out_c = torch.log_softmax(self.policy.critic(meta_v_tensor, samples, stop_meta_v_emb_grad=True), dim=1)
                else:
                    out_a = torch.log_softmax(self.policy.actor(meta_v_tensor, samples, stop_meta_v_emb_grad=True, pos_emb=pos_emb_tensor), dim=1)
                    out_c = torch.log_softmax(self.policy.critic(meta_v_tensor, samples, stop_meta_v_emb_grad=True, pos_emb=pos_emb_tensor), dim=1)
                loss = self.KLDivLoss(out_a, label_a) + self.KLDivLoss(out_c, label_c)
            elif loss_type == 'mse':
                with torch.no_grad():
                    label_a = ppo.policy.actor(samples)
                    label_c = ppo.policy.critic(samples)
                if not self.pos_emb:
                    if not self.meta_v_emb:
                        out_a, _ = self.policy.actor(meta_v_tensor, samples)
                        out_c, _ = self.policy.critic(meta_v_tensor, samples)
                    else:
                        out_a, _ = self.policy.actor(meta_v_tensor, samples, stop_meta_v_emb_grad=True)
                        out_c, _ = self.policy.critic(meta_v_tensor, samples, stop_meta_v_emb_grad=True)
                else:
                    out_a, _ = self.policy.actor(meta_v_tensor, samples, stop_meta_v_emb_grad=True, pos_emb=pos_emb_tensor)
                    out_c, _ = self.policy.critic(meta_v_tensor, samples, stop_meta_v_emb_grad=True, pos_emb=pos_emb_tensor)
                loss = F.mse_loss(out_a, label_a) + F.mse_loss(out_c, label_c)
            else:
                raise ValueError(f'Unknown loss type {loss_type}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('pretrain_loss', loss.detach().item(), i)
            if i % 1000 == 0:
                print("(Pret)Seed:{}, env:{}, Step:#{}/{}, Loss:{:.6f}, Type:{}, Pol:{}, N:{}-{}, Info:{}"
                      .format(seed, env.env_name, i, pretrain_step, loss.detach().item(), loss_type,
                              'unified_a{}_c{}'.format(int(self.hyper_actor), int(self.hyper_critic)),
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
                action, action_logprob, action_prob = self.policy_old.act(meta_v, state, ret_prob, pos_emb=pos_emb)
            else:
                action, action_logprob = self.policy_old.act(meta_v, state, pos_emb=pos_emb)
        if store_tuple:
            self.buffer.meta_vs.append(meta_v[store_tuple_idx])
            self.buffer.states.append(state[store_tuple_idx])
            self.buffer.actions.append(action[store_tuple_idx])
            self.buffer.logprobs.append(action_logprob[store_tuple_idx])
            if pos_emb is not None:
                self.buffer.pos_emb.append(pos_emb[store_tuple_idx])

        if ret_prob:
            return action.detach().cpu().numpy(), action_prob.detach().cpu().numpy()

        return action.detach().cpu().numpy()

    def select_action_finetune(self, meta_v, state, store_tuple=True, store_tuple_idx=0, ret_prob=False, pos_emb=None):
        with torch.no_grad():
            state = torch.FloatTensor(state)
            if self.cuda:
                state = state.to(self.device)
            action, action_logprob = self.policy_finetune_old.act(meta_v, state)
        if store_tuple:
            self.buffer.meta_vs.append(meta_v[store_tuple_idx])
            self.buffer.states.append(state[store_tuple_idx])
            self.buffer.actions.append(action[store_tuple_idx])
            self.buffer.logprobs.append(action_logprob[store_tuple_idx])
            if pos_emb is not None:
                self.buffer.pos_emb.append(pos_emb[store_tuple_idx])

        return action.detach().cpu().numpy()

    def update(self):
        # Monte Carlo estimate of returns
        # input('FDSFSFDSFSFDSFSFSFSF=============')
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
        if self.meta_v_dim == 1:
            old_meta_vs = old_meta_vs.unsqueeze(1)
        # print(old_meta_vs[50:70])
        # print(old_states.shape)
        # print(old_actions.shape)
        # print(old_logprobs.shape)
        # input('dddddd')
        if self.pos_emb:
            old_pos_embs = torch.squeeze(torch.stack(self.buffer.pos_emb, dim=0)).detach()
            if self.cuda:
                old_pos_embs = old_pos_embs.to(self.device)
        else:
            old_pos_embs = None

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_meta_vs, old_states, old_actions, pos_emb=old_pos_embs)

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
                if self.l1: # l1 regularization
                    # l1_loss = l1_reg_loss(self.policy)
                    # print('main loss: {:.8f}, l1 loss: {:.8f}'.format(loss.item(), 1e-6 * l1_loss.item())) # main loss: 0.01~0.2, l1 loss: 40000~50000
                    loss = loss + 1e-6 * l1_reg_loss(self.policy)

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

    def update_finetune(self):
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
        if self.meta_v_dim == 1:
            old_meta_vs = old_meta_vs.unsqueeze(1)
        # print(old_meta_vs[50:70])
        # print(old_states.shape)
        # print(old_actions.shape)
        # print(old_logprobs.shape)
        # input('dddddd')
        if self.pos_emb:
            old_pos_embs = torch.squeeze(torch.stack(self.buffer.pos_emb, dim=0)).detach()
            if self.cuda:
                old_pos_embs = old_pos_embs.to(self.device)
        else:
            old_pos_embs = None

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy_finetune.evaluate(old_meta_vs, old_states, old_actions)

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
                if self.l1: # l1 regularization
                    # l1_loss = l1_reg_loss(self.policy)
                    # print('main loss: {:.8f}, l1 loss: {:.8f}'.format(loss.item(), 1e-6 * l1_loss.item())) # main loss: 0.01~0.2, l1 loss: 40000~50000
                    loss = loss + 1e-6 * l1_reg_loss(self.policy_finetune)

            # take gradient step
            self.optimizer_finetune.zero_grad()
            loss.backward()
            if self.w_clip:  # Weight Clip
                nn.utils.clip_grad_norm_(self.policy_finetune.parameters(), self.w_clip_p)
            self.optimizer_finetune.step()
            # print(self.policy.actor.input_layer.W1.weight.grad)

        # Copy new weights into old policy
        self.policy_finetune_old.load_state_dict(self.policy_finetune.state_dict())
        # print('after: ', self.policy_finetune.actor.input_layer_w.data[0][0][:50])
        # self.policy_finetune_old.actor.init_emb_layer(self.policy_finetune.actor.meta_v_emb_layer)
        # self.policy_finetune_old.critic.init_emb_layer(self.policy_finetune.critic.meta_v_emb_layer)
        # self.policy_finetune_old.actor.init_network([self.policy_finetune.actor.input_layer_w, self.policy_finetune.actor.hidden_layer_w, self.policy_finetune.actor.output_layer_w],
        #                                             [self.policy_finetune.actor.input_layer_b, self.policy_finetune.actor.hidden_layer_b, self.policy_finetune.actor.output_layer_b],
        #                                             [self.policy_finetune.actor.input_layer_s, self.policy_finetune.actor.hidden_layer_s, self.policy_finetune.actor.output_layer_s])
        # self.policy_finetune_old.critic.init_network([self.policy_finetune.critic.input_layer_w, self.policy_finetune.critic.hidden_layer_w, self.policy_finetune.critic.output_layer_w],
        #                                              [self.policy_finetune.critic.input_layer_b, self.policy_finetune.critic.hidden_layer_b, self.policy_finetune.critic.output_layer_b],
        #                                              [self.policy_finetune.critic.input_layer_s, self.policy_finetune.critic.hidden_layer_s, self.policy_finetune.critic.output_layer_s])

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

    def save_finetune(self, checkpoint_path):
        torch.save(self.policy_finetune_old.state_dict(), checkpoint_path)

    def load_finetune(self, checkpoint_path):
        self.policy_finetune_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy_finetune.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))