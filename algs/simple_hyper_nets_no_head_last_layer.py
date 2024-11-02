import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_weights_kaiming_uniform


#========== 2 Layers ==========================
class HyperNetwork2Layer(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, output_dim, hyper_hidden_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork2Layer, self).__init__()
        self.meta_v_dim = meta_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.last_layer_w = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * output_dim)
                                          )
        self.last_layer_b = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, output_dim)
                                          )
        self.init_layers()

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def forward(self, meta_v):
        w, b = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
               self.last_layer_b(meta_v).view(-1, self.output_dim, 1)
        return w, b


class HyperNetworkEmbedInput2Layer(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, output_dim, hyper_hidden_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput2Layer, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb_dim = meta_v_emb_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.last_layer_w = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * output_dim)
                                          )
        self.last_layer_b = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, output_dim)
                                          )
        self.init_layers()
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def forward(self, meta_v, pos_emb=None, scale=True):
        emb = self.meta_v_emb_layer(meta_v)
        if self.clip_emb:
            emb = torch.tanh(emb)
        if pos_emb is not None:
            if scale:
               emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
            else:
                emb = pos_emb + emb
        w, b = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
               self.last_layer_b(emb).view(-1, self.output_dim, 1)
        return w, b, emb