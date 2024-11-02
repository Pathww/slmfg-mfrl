import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_weights_kaiming_uniform


class Head(nn.Module):
    def __init__(self, latent_dim, output_dim_in, output_dim_out, sttdev):
        super(Head, self).__init__()
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out
        self.W1 = nn.Linear(latent_dim, output_dim_in * output_dim_out)
        self.b1 = nn.Linear(latent_dim, output_dim_out)
        self.s1 = nn.Linear(latent_dim, output_dim_out)
        self.init_layers(sttdev)

    def forward(self, x, ret_s_original=False):
        w = self.W1(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).view(-1, self.output_dim_out, 1)

        if ret_s_original:
            s = self.s1(x).view(-1, self.output_dim_out, 1)
            return w, b, s
        else:
            s = 1.0 + self.s1(x).view(-1, self.output_dim_out, 1)
            return w, b, s

    def init_layers(self, stddev):
        torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.s1.weight, -stddev, stddev)
        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.zeros_(self.b1.bias)
        torch.nn.init.zeros_(self.s1.bias)


#========== 1 Layers ==========================
class Meta_Embadding1Layer(nn.Module):
    def __init__(self, meta_dim, z_dim=128):
        super(Meta_Embadding1Layer, self).__init__()
        self.z_dim = z_dim
        self.hyper = nn.Sequential(
            nn.Linear(meta_dim, z_dim),
        )
        self.init_layers()

    def forward(self, meta_v):
        z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)


class HyperNetwork1Layer(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, z_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork1Layer, self).__init__()
        self.hyper = Meta_Embadding1Layer(meta_v_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInput1Layer(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, z_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput1Layer, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding1Layer(meta_v_emb_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True, bp_emb='hyper'):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach() * (self.meta_v_emb_dim ** 0.5)
                else:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach()
            else:
                if bp_emb == 'hyper':
                    pos_plus_emb = emb
                else:
                    pos_plus_emb = emb.clone().detach()
            z = self.hyper(pos_plus_emb)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        if bp_emb == 'hyper':
            base_v = torch.cat([base_v, emb.clone().detach()], dim=1)
        else:
            base_v = torch.cat([base_v, emb], dim=1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetwork1LayerCH(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, z_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork1LayerCH, self).__init__()
        self.hyper = Meta_Embadding1Layer(meta_v_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim + meta_v_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, torch.cat([out, meta_v.unsqueeze(2)], dim=1)) * s2 + b2)
        out = torch.bmm(w3, torch.cat([out, meta_v.unsqueeze(2)], dim=1)) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, torch.cat([out_1, meta_v.unsqueeze(2)], dim=1)) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, torch.cat([out_3, meta_v.unsqueeze(2)], dim=1)) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInput1LayerCH(nn.Module):
    # meta_v --> embedding --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, z_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput1LayerCH, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding1Layer(meta_v_emb_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim + meta_v_emb_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True, bp_emb='hyper'):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach() * (self.meta_v_emb_dim ** 0.5)
                else:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach()
            else:
                if bp_emb == 'hyper':
                    pos_plus_emb = emb
                else:
                    pos_plus_emb = emb.clone().detach()
            z = self.hyper(pos_plus_emb)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        if bp_emb == 'hyper':
            base_v = torch.cat([base_v, emb.clone().detach()], dim=1)
        else:
            base_v = torch.cat([base_v, emb], dim=1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        if stop_meta_v_emb_grad:
            out = F.relu(torch.bmm(w2, torch.cat([out, emb.unsqueeze(2)], dim=1)) * s2 + b2)
            out = torch.bmm(w3, torch.cat([out, emb.unsqueeze(2)], dim=1)) * s3 + b3
        else:
            out = F.relu(torch.bmm(w2, torch.cat([out, emb.clone().detach().unsqueeze(2)], dim=1)) * s2 + b2)
            out = torch.bmm(w3, torch.cat([out, emb.clone().detach().unsqueeze(2)], dim=1)) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, torch.cat([out_1, emb.unsqueeze(2)])) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, torch.cat([out_3, emb.unsqueeze(2)])) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetwork1LayerNoAug(nn.Module):
    # meta_v --> hypernet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, z_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork1LayerNoAug, self).__init__()
        self.hyper = Meta_Embadding1Layer(meta_v_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInput1LayerNoAug(nn.Module):
    # meta_v --> embedding --> hypernet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, z_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput1LayerNoAug, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding1Layer(meta_v_emb_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


#========== 2 Layers ==========================
class Meta_Embadding2Layer(nn.Module):
    def __init__(self, meta_dim, hidden_dim=128, z_dim=128):
        super(Meta_Embadding2Layer, self).__init__()
        self.z_dim = z_dim
        self.hyper = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        self.init_layers()

    def forward(self, meta_v):
        z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)


class HyperNetwork2Layer(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork2Layer, self).__init__()
        self.hyper = Meta_Embadding2Layer(meta_v_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInput2Layer(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput2Layer, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding2Layer(meta_v_emb_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True, bp_emb='hyper'):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if bp_emb == 'hyper':
                z = self.hyper(emb)
            elif bp_emb == 'all':
                # print('dddddddd')
                # input('fdsafasdfasfdfas')
                z = self.hyper(emb)
            else:
                z = self.hyper(emb.clone().detach())
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        # print(meta_v.shape) # [100, 12]
        # print(w1.shape)     # [100, 128, 171]
        # print(b1.shape)     # [100, 128, 1]
        # print(s1.shape)     # [100, 128, 1]
        # print(w2.shape)     # [100, 128, 128]
        # print(b2.shape)     # [100, 128, 1]
        # print(s2.shape)     # [100, 128, 1]
        # print(w3.shape)     # [100, 5, 128]
        # print(b3.shape)     # [100, 5, 1]
        # print(s3.shape)     # [100, 5, 1]
        # print(base_v.shape) # [100, state_dim]
        # print(w1[:2, 0, :2])
        # input('AAAAAAAA')
        if bp_emb == 'hyper':
            base_v = torch.cat([base_v, emb.clone().detach()], dim=1)
        elif bp_emb == 'all':
            # input('fdsafasdfasfdfas')
            base_v = torch.cat([base_v, emb], dim=1)
        else:
            base_v = torch.cat([base_v, emb], dim=1)
        # print(meta_v)
        # aa = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
        # print(aa.shape)
        # print(aa[:2, 0, :])
        # print(s1.shape)
        # print(s1[0, :2, :])
        # input('AAAA')

        # state: [batch_size, state_dim]
        # embedding: [batch_size, embedding_dim]
        # w1: [batch_size, 128, state_dim + embedding_dim], b1: [batch_size, 128, 1],          g1: [batch_size, 128, 1]
        # w2: [batch_size, 128, 128],                       b2: [batch_size, 128, 1],          g2: [batch_size, 128, 1]
        # w3: [batch_size, action_dim, 128],                b3: [batch_size, action_dim, 1],   g3: [batch_size, action_dim, 1]
        # out = F.relu(torch.bmm(w1, torch.cat([state, embedding], dim=1).unsqueeze(2)) * s1 + b1)
        # out = F.relu(torch.bmm(w2, out) * s2 + b2)
        # out = torch.bmm(w3, out) * s3 + b3
        # logits = torch.squeeze(out, dim=2)

        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        # print(out[:4, 0, :])
        # input('AAAA')
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        # print(out[:4, 0, :])
        # input('BBBB')
        out = torch.bmm(w3, out) * s3 + b3
        # print(out[:4, 0, :])
        # input('CCCC')
        out = torch.squeeze(out, dim=2)
        return out
        # if stop_meta_v_emb_grad:
        #     with torch.no_grad():
        #         emb = self.meta_v_emb_layer(meta_v)
        #         if self.clip_emb:
        #             emb = torch.tanh(emb)
        #         if pos_emb is not None:
        #             if scale:
        #                 pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
        #             else:
        #                 pos_plus_emb = pos_emb + emb
        #         else:
        #             pos_plus_emb = emb
        #     z = self.hyper(pos_plus_emb)
        # else:
        #     emb = self.meta_v_emb_layer(meta_v)
        #     if self.clip_emb:
        #         emb = torch.tanh(emb)
        #     if pos_emb is not None:
        #         if scale:
        #             if bp_emb == 'hyper':
        #                 pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
        #             else:
        #                 pos_plus_emb = pos_emb + emb.clone().detach() * (self.meta_v_emb_dim ** 0.5)
        #         else:
        #             if bp_emb == 'hyper':
        #                 pos_plus_emb = pos_emb + emb
        #             else:
        #                 pos_plus_emb = pos_emb + emb.clone().detach()
        #     else:
        #         if bp_emb == 'hyper':
        #             pos_plus_emb = emb
        #         elif bp_emb == 'all':
        #             pos_plus_emb = emb
        #         else:
        #             pos_plus_emb = emb.clone().detach()
        #     z = self.hyper(pos_plus_emb)
        # w1, b1, s1 = self.input_layer(z)
        # w2, b2, s2 = self.hidden(z)
        # w3, b3, s3 = self.last_layer(z)
        # if bp_emb == 'hyper':
        #     base_v = torch.cat([base_v, emb.clone().detach()], dim=1)
        # elif bp_emb == 'all':
        #     base_v = torch.cat([base_v, emb], dim=1)
        # else:
        #     base_v = torch.cat([base_v, emb], dim=1)
        # out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        # out = F.relu(torch.bmm(w2, out) * s2 + b2)
        # out = torch.bmm(w3, out) * s3 + b3
        # out = torch.squeeze(out, dim=2)
        # return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetwork2LayerCH(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork2LayerCH, self).__init__()
        self.hyper = Meta_Embadding2Layer(meta_v_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim + meta_v_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, torch.cat([out, meta_v.unsqueeze(2)], dim=1)) * s2 + b2)
        out = torch.bmm(w3, torch.cat([out, meta_v.unsqueeze(2)], dim=1)) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, torch.cat([out_1, meta_v.unsqueeze(2)], dim=1)) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, torch.cat([out_3, meta_v.unsqueeze(2)], dim=1)) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInput2LayerCH(nn.Module):
    # meta_v --> embedding --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput2LayerCH, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding2Layer(meta_v_emb_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim + meta_v_emb_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True, bp_emb='hyper', relu_emb_in_hidden=False):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach() * (self.meta_v_emb_dim ** 0.5)
                else:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach()
            else:
                if bp_emb == 'hyper':
                    pos_plus_emb = emb
                elif bp_emb == 'all':
                    pos_plus_emb = emb
                else:
                    pos_plus_emb = emb.clone().detach()
            z = self.hyper(pos_plus_emb)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        if bp_emb == 'hyper':
            base_v = torch.cat([base_v, emb.clone().detach()], dim=1)
        elif bp_emb == 'all':
            base_v = torch.cat([base_v, emb], dim=1)
        else:
            base_v = torch.cat([base_v, emb], dim=1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        if stop_meta_v_emb_grad:
            if relu_emb_in_hidden:
                out = F.relu(torch.bmm(w2, torch.cat([out, F.relu(emb).unsqueeze(2)], dim=1)) * s2 + b2)
                out = torch.bmm(w3, torch.cat([out, F.relu(emb).unsqueeze(2)], dim=1)) * s3 + b3
            else:
                out = F.relu(torch.bmm(w2, torch.cat([out, emb.unsqueeze(2)], dim=1)) * s2 + b2)
                out = torch.bmm(w3, torch.cat([out, emb.unsqueeze(2)], dim=1)) * s3 + b3
        else:
            if relu_emb_in_hidden:
                out = F.relu(torch.bmm(w2, torch.cat([out, F.relu(emb.clone().detach()).unsqueeze(2)], dim=1)) * s2 + b2)
                out = torch.bmm(w3, torch.cat([out, F.relu(emb.clone().detach()).unsqueeze(2)], dim=1)) * s3 + b3
            else:
                out = F.relu(torch.bmm(w2, torch.cat([out, emb.clone().detach().unsqueeze(2)], dim=1)) * s2 + b2)
                out = torch.bmm(w3, torch.cat([out, emb.clone().detach().unsqueeze(2)], dim=1)) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, relu_emb_in_hidden=False, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            if relu_emb_in_hidden:
                out_2 = torch.bmm(w2, torch.cat([out_1, F.relu(emb).unsqueeze(2)])) * s2 + b2
            else:
                out_2 = torch.bmm(w2, torch.cat([out_1, emb.unsqueeze(2)])) * s2 + b2
            out_3 = F.relu(out_2)
            if relu_emb_in_hidden:
                out_4 = torch.bmm(w3, torch.cat([out_3, F.relu(emb).unsqueeze(2)])) * s3 + b3
            else:
                out_4 = torch.bmm(w3, torch.cat([out_3, emb.unsqueeze(2)])) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetwork2LayerNoAug(nn.Module):
    # meta_v --> hypernet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128):
        super(HyperNetwork2LayerNoAug, self).__init__()
        self.hyper = Meta_Embadding2Layer(meta_v_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInput2LayerNoAug(nn.Module):
    # meta_v --> embedding --> hypernet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hyper_hidden_dim=128, z_dim=128, dynamic_hidden_dim=128, clip_emb=False):
        super(HyperNetworkEmbedInput2LayerNoAug, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding2Layer(meta_v_emb_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


#========== 4 Layers ==========================
class Meta_Embadding4Layer(nn.Module):
    def __init__(self, meta_dim, hidden_dim=256, z_dim=256):
        super(Meta_Embadding4Layer, self).__init__()
        self.z_dim = z_dim
        self.hyper = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim)
        )
        self.init_layers()

    def forward(self, meta_v):
        z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):
        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)


class HyperNetwork(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256):
        super(HyperNetwork, self).__init__()
        self.hyper = Meta_Embadding4Layer(meta_v_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInput(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256, clip_emb=False):
        super(HyperNetworkEmbedInput, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding4Layer(meta_v_emb_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True, bp_emb='hyper'):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach() * (self.meta_v_emb_dim ** 0.5)
                else:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach()
            else:
                if bp_emb == 'hyper':
                    pos_plus_emb = emb
                else:
                    pos_plus_emb = emb.clone().detach()
            z = self.hyper(pos_plus_emb)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        if bp_emb == 'hyper':
            base_v = torch.cat([base_v, emb.clone().detach()], dim=1)
        else:
            base_v = torch.cat([base_v, emb], dim=1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkCH(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256):
        super(HyperNetworkCH, self).__init__()
        self.hyper = Meta_Embadding4Layer(meta_v_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim + meta_v_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim + meta_v_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, torch.cat([out, meta_v.unsqueeze(2)], dim=1)) * s2 + b2)
        out = torch.bmm(w3, torch.cat([out, meta_v.unsqueeze(2)], dim=1)) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, torch.cat([out_1, meta_v.unsqueeze(2)], dim=1)) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, torch.cat([out_3, meta_v.unsqueeze(2)], dim=1)) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInputCH(nn.Module):
    # meta_v --> embedding --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256, clip_emb=False):
        super(HyperNetworkEmbedInputCH, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding4Layer(meta_v_emb_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim + meta_v_emb_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim + meta_v_emb_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True, bp_emb='hyper'):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach() * (self.meta_v_emb_dim ** 0.5)
                else:
                    if bp_emb == 'hyper':
                        pos_plus_emb = pos_emb + emb
                    else:
                        pos_plus_emb = pos_emb + emb.clone().detach()
            else:
                if bp_emb == 'hyper':
                    pos_plus_emb = emb
                else:
                    pos_plus_emb = emb.clone().detach()
            z = self.hyper(pos_plus_emb)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        if bp_emb == 'hyper':
            base_v = torch.cat([base_v, emb.clone().detach()], dim=1)
        else:
            base_v = torch.cat([base_v, emb], dim=1)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        if stop_meta_v_emb_grad:
            out = F.relu(torch.bmm(w2, torch.cat([out, emb.unsqueeze(2)], dim=1)) * s2 + b2)
            out = torch.bmm(w3, torch.cat([out, emb.unsqueeze(2)], dim=1)) * s3 + b3
        else:
            out = F.relu(torch.bmm(w2, torch.cat([out, emb.clone().detach().unsqueeze(2)], dim=1)) * s2 + b2)
            out = torch.bmm(w3, torch.cat([out, emb.clone().detach().unsqueeze(2)], dim=1)) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, torch.cat([out_1, emb.unsqueeze(2)])) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, torch.cat([out_3, emb.unsqueeze(2)])) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkNoAug(nn.Module):
    # meta_v --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256):
        super(HyperNetworkNoAug, self).__init__()
        self.hyper = Meta_Embadding4Layer(meta_v_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v):
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, before_relu=False):
        with torch.no_grad():
            z = self.hyper(meta_v)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]


class HyperNetworkEmbedInputNoAug(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hyper_hidden_dim=256, z_dim=256, dynamic_hidden_dim=256, clip_emb=False):
        super(HyperNetworkEmbedInputNoAug, self).__init__()
        self.clip_emb = clip_emb
        self.meta_v_emb_dim = meta_v_emb_dim
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        self.hyper = Meta_Embadding4Layer(meta_v_emb_dim, hyper_hidden_dim, z_dim)
        self.input_layer = Head(z_dim, base_v_dim, dynamic_hidden_dim, sttdev=0.05)
        self.hidden = Head(z_dim, dynamic_hidden_dim, dynamic_hidden_dim, sttdev=0.008)
        self.last_layer = Head(z_dim, dynamic_hidden_dim, output_dim, sttdev=0.001)

    def get_weight(self, meta_v, no_grad=True, pos_emb=None, scale=True, keep_shape=False):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
                z = self.hyper(pos_plus_emb)
                w1, b1, s1 = self.input_layer(z, ret_s_original=True)
                w2, b2, s2 = self.hidden(z, ret_s_original=True)
                w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        else:
            emb = self.meta_v_emb_layer(meta_v).detach()
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z, ret_s_original=True)
            w2, b2, s2 = self.hidden(z, ret_s_original=True)
            w3, b3, s3 = self.last_layer(z, ret_s_original=True)
        if not keep_shape:
            return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
                   [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)], \
                   [torch.flatten(s1, start_dim=1), torch.flatten(s2, start_dim=1), torch.flatten(s3, start_dim=1)]
        else:
            return [w1, w2, w3], [b1, b2, b3], [s1, s2, s3]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None, scale=True):
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = torch.tanh(emb)
                if pos_emb is not None:
                    if scale:
                        pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                    else:
                        pos_plus_emb = pos_emb + emb
                else:
                    pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None, scale=True, before_relu=False):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = torch.tanh(emb)
            if pos_emb is not None:
                if scale:
                    pos_plus_emb = pos_emb + emb * (self.meta_v_emb_dim ** 0.5)
                else:
                    pos_plus_emb = pos_emb + emb
            else:
                pos_plus_emb = emb
            z = self.hyper(pos_plus_emb)
            w1, b1, s1 = self.input_layer(z)
            w2, b2, s2 = self.hidden(z)
            w3, b3, s3 = self.last_layer(z)
            out_0 = torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) * s2 + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) * s3 + b3
            out_4 = torch.squeeze(out_4, dim=2)
        if before_relu:
            return [out_0, out_2, out_4]
        else:
            return [out_1, out_3, out_4]