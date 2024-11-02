import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_weights_kaiming_uniform


#========== 1 Layers ==========================
class HyperNetwork1Layer(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, dynamic_hidden_dim=128):
        super(HyperNetwork1Layer, self).__init__()
        self.meta_v_dim = meta_v_dim
        self.base_v_dim = base_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.input_layer_w = nn.Linear(meta_v_dim, (base_v_dim + meta_v_dim) * dynamic_hidden_dim)
        self.input_layer_b = nn.Linear(meta_v_dim, dynamic_hidden_dim)
        self.input_layer_s = nn.Linear(meta_v_dim, dynamic_hidden_dim)
        self.hidden_layer_w = nn.Linear(meta_v_dim, dynamic_hidden_dim * dynamic_hidden_dim)
        self.hidden_layer_b = nn.Linear(meta_v_dim, dynamic_hidden_dim)
        self.hidden_layer_s = nn.Linear(meta_v_dim, dynamic_hidden_dim)
        self.last_layer_w = nn.Linear(meta_v_dim, dynamic_hidden_dim * output_dim)
        self.last_layer_b = nn.Linear(meta_v_dim, output_dim)
        self.last_layer_s = nn.Linear(meta_v_dim, output_dim)
        self.init_layers()

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def get_weight(self, meta_v):
        with torch.no_grad():
            w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
               [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]

    def forward(self, meta_v, base_v):
        w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                     self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
        w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim),\
                     self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
        w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                     self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                     1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v):
        with torch.no_grad():
            w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetworkEmbedInput1Layer(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput1Layer, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb_dim = meta_v_emb_dim
        self.base_v_dim = base_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.input_layer_w = nn.Linear(meta_v_emb_dim, (base_v_dim + meta_v_emb_dim) * dynamic_hidden_dim)
        self.input_layer_b = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim)
        self.input_layer_s = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim)
        self.hidden_layer_w = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim * dynamic_hidden_dim)
        self.hidden_layer_b = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim)
        self.hidden_layer_s = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim)
        self.last_layer_w = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim * output_dim)
        self.last_layer_b = nn.Linear(meta_v_emb_dim, output_dim)
        self.last_layer_s = nn.Linear(meta_v_emb_dim, output_dim)
        self.init_layers()
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def get_weight(self, meta_v, pos_emb=None, scale=True):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
            w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_emb_dim), \
                         self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
               [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        emb = pos_emb + emb
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
        w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_emb_dim), \
                     self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
        w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                     self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
        w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                     self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                     1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
            w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetwork1LayerNoAug(nn.Module):
    # meta_v --> hypernet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, dynamic_hidden_dim=128):
        super(HyperNetwork1LayerNoAug, self).__init__()
        self.meta_v_dim = meta_v_dim
        self.base_v_dim = base_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.input_layer_w = nn.Linear(meta_v_dim, base_v_dim * dynamic_hidden_dim)
        self.input_layer_b = nn.Linear(meta_v_dim, dynamic_hidden_dim)
        self.input_layer_s = nn.Linear(meta_v_dim, dynamic_hidden_dim)
        self.hidden_layer_w = nn.Linear(meta_v_dim, dynamic_hidden_dim * dynamic_hidden_dim)
        self.hidden_layer_b = nn.Linear(meta_v_dim, dynamic_hidden_dim)
        self.hidden_layer_s = nn.Linear(meta_v_dim, dynamic_hidden_dim)
        self.last_layer_w = nn.Linear(meta_v_dim, dynamic_hidden_dim * output_dim)
        self.last_layer_b = nn.Linear(meta_v_dim, output_dim)
        self.last_layer_s = nn.Linear(meta_v_dim, output_dim)
        self.init_layers()

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def get_weight(self, meta_v):
        with torch.no_grad():
            w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
               [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]

    def forward(self, meta_v, base_v):
        w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim), \
                     self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
        w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim),\
                     self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
        w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                     self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                     1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v):
        with torch.no_grad():
            w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetworkEmbedInput1LayerNoAug(nn.Module):
    # meta_v --> embedding --> hypernet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput1LayerNoAug, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb_dim = meta_v_emb_dim
        self.base_v_dim = base_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.input_layer_w = nn.Linear(meta_v_emb_dim, base_v_dim * dynamic_hidden_dim)
        self.input_layer_b = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim)
        self.input_layer_s = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim)
        self.hidden_layer_w = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim * dynamic_hidden_dim)
        self.hidden_layer_b = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim)
        self.hidden_layer_s = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim)
        self.last_layer_w = nn.Linear(meta_v_emb_dim, dynamic_hidden_dim * output_dim)
        self.last_layer_b = nn.Linear(meta_v_emb_dim, output_dim)
        self.last_layer_s = nn.Linear(meta_v_emb_dim, output_dim)
        self.init_layers()
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def get_weight(self, meta_v, pos_emb=None, scale=True):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
            w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_emb_dim), \
                         self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
               [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        emb = pos_emb + emb
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
        w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim), \
                     self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
        w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                     self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
        w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                     self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                     1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
            w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


#========== 2 Layers ==========================
class HyperNetwork2Layer(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hyper_hidden_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork2Layer, self).__init__()
        self.meta_v_dim = meta_v_dim
        self.base_v_dim = base_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.input_layer_w = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, (base_v_dim + meta_v_dim) * dynamic_hidden_dim)
                                           )
        self.input_layer_b = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                           )
        self.input_layer_s = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                           )
        self.hidden_layer_w = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * dynamic_hidden_dim)
                                            )
        self.hidden_layer_b = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                            )
        self.hidden_layer_s = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                            )
        self.last_layer_w = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * output_dim)
                                          )
        self.last_layer_b = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, output_dim)
                                          )
        self.last_layer_s = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, output_dim)
                                          )
        # self.apply(init_weights_kaiming_uniform)
        self.init_layers()

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def get_weight(self, meta_v):
        with torch.no_grad():
            w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
               [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]

    def forward(self, meta_v, base_v):
        w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                     self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
        w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim),\
                     self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
        w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                     self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                     1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v):
        with torch.no_grad():
            w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetworkEmbedInput2Layer(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hyper_hidden_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput2Layer, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb_dim = meta_v_emb_dim
        self.base_v_dim = base_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.input_layer_w = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, (base_v_dim + meta_v_emb_dim) * dynamic_hidden_dim)
                                           )
        self.input_layer_b = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                           )
        self.input_layer_s = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                           )
        self.hidden_layer_w = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * dynamic_hidden_dim)
                                            )
        self.hidden_layer_b = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                            )
        self.hidden_layer_s = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                            )
        self.last_layer_w = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * output_dim)
                                          )
        self.last_layer_b = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, output_dim)
                                          )
        self.last_layer_s = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
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

    def get_weight(self, meta_v, pos_emb=None, scale=True):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
            w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_emb_dim), \
                         self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
               [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        emb = pos_emb + emb
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                   emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
        w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_emb_dim), \
                     self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
        w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                     self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
        w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                     self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                     1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
            w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetwork2LayerNoAug(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hyper_hidden_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork2LayerNoAug, self).__init__()
        self.meta_v_dim = meta_v_dim
        self.base_v_dim = base_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.input_layer_w = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, base_v_dim * dynamic_hidden_dim)
                                           )
        self.input_layer_b = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                           )
        self.input_layer_s = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                           )
        self.hidden_layer_w = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * dynamic_hidden_dim)
                                            )
        self.hidden_layer_b = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                            )
        self.hidden_layer_s = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                            )
        self.last_layer_w = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * output_dim)
                                          )
        self.last_layer_b = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, output_dim)
                                          )
        self.last_layer_s = nn.Sequential(nn.Linear(meta_v_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, output_dim)
                                          )
        # self.apply(init_weights_kaiming_uniform)
        self.init_layers()

    def init_layers(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)

    def get_weight(self, meta_v):
        with torch.no_grad():
            w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
               [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]

    def forward(self, meta_v, base_v):
        w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim), \
                     self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
        w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim),\
                     self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
        w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                     self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                     1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v):
        with torch.no_grad():
            w1, b1, s1 = self.input_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(meta_v).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(meta_v).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(meta_v).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(meta_v).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(meta_v).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(meta_v).view(-1, self.output_dim, 1)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetworkEmbedInput2LayerNoAug(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hyper_hidden_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput2LayerNoAug, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_dim = meta_v_dim
        self.meta_v_emb_dim = meta_v_emb_dim
        self.base_v_dim = base_v_dim
        self.dynamic_hidden_dim = dynamic_hidden_dim
        self.output_dim = output_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.input_layer_w = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, base_v_dim * dynamic_hidden_dim)
                                           )
        self.input_layer_b = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                           )
        self.input_layer_s = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                           nn.ReLU(),
                                           nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                           )
        self.hidden_layer_w = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * dynamic_hidden_dim)
                                            )
        self.hidden_layer_b = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                            )
        self.hidden_layer_s = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hyper_hidden_dim, dynamic_hidden_dim)
                                            )
        self.last_layer_w = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, dynamic_hidden_dim * output_dim)
                                          )
        self.last_layer_b = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hyper_hidden_dim, output_dim)
                                          )
        self.last_layer_s = nn.Sequential(nn.Linear(meta_v_emb_dim, hyper_hidden_dim),
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

    def get_weight(self, meta_v, pos_emb=None, scale=True):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
            w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_emb_dim), \
                         self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
               [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        emb = pos_emb + emb
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                   emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
        w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim), \
                     self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
        w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                     self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                     1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
        w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                     self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                     1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    emb = pos_emb + emb
            w1, b1, s1 = self.input_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.base_v_dim + self.meta_v_dim), \
                         self.input_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.input_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w2, b2, s2 = self.hidden_layer_w(emb).view(-1, self.dynamic_hidden_dim, self.dynamic_hidden_dim), \
                         self.hidden_layer_b(emb).view(-1, self.dynamic_hidden_dim, 1), \
                         1.0 + self.hidden_layer_s(emb).view(-1, self.dynamic_hidden_dim, 1)
            w3, b3, s3 = self.last_layer_w(emb).view(-1, self.output_dim, self.dynamic_hidden_dim), \
                         self.last_layer_b(emb).view(-1, self.output_dim, 1), \
                         1.0 + self.last_layer_s(emb).view(-1, self.output_dim, 1)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]