import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init_weights_kaiming_uniform


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(ResBlock, self).__init__()

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_size, out_size),
            nn.ReLU(),
            nn.Linear(out_size, out_size),
        )

    def forward(self, x):
        h = self.fc(x)
        return x + h


class Head(nn.Module):
    def __init__(self, latent_dim, output_dim_in, output_dim_out, sttdev):
        super(Head, self).__init__()

        h_layer = latent_dim
        self.output_dim_in = output_dim_in
        self.output_dim_out = output_dim_out

        self.W1 = nn.Linear(h_layer, output_dim_in * output_dim_out)
        self.b1 = nn.Linear(h_layer, output_dim_out)
        # self.s1 = nn.Linear(h_layer, output_dim_out)

        self.init_layers(sttdev)

    def forward(self, x):
        # weights, bias and scale for dynamic layer
        w = self.W1(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).view(-1, self.output_dim_out, 1)
        # s = 1.0 + self.s1(x).view(-1, self.output_dim_out, 1)

        return w, b#, s

    def init_layers(self, stddev):
        torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)
        # torch.nn.init.uniform_(self.s1.weight, -stddev, stddev)

        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.zeros_(self.b1.bias)
        # torch.nn.init.zeros_(self.s1.bias)


class Meta_Embadding(nn.Module):
    def __init__(self, meta_dim, z_dim):
        super(Meta_Embadding, self).__init__()

        self.z_dim = z_dim

        self.hyper = nn.Sequential(
            nn.Linear(meta_dim, 256),
            ResBlock(256, 256),
            nn.Linear(256, 512),
            ResBlock(512, 512),
            nn.Linear(512, 1024),
            ResBlock(1024, 1024),
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
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hidden_layer=256, hyper_emb_dim=1024):
        super(HyperNetwork, self).__init__()
        z_dim = hyper_emb_dim
        self._hidden_layer = hidden_layer
        # encode the meta_v into a hidden embedding
        self.hyper = Meta_Embadding(meta_v_dim, z_dim)
        # Q function net, use the hidden embedding to generate weights
        self.input_layer = Head(z_dim, base_v_dim + meta_v_dim, hidden_layer, sttdev=0.05)
        self.hidden = Head(z_dim, hidden_layer, hidden_layer, sttdev=0.008)
        self.last_layer = Head(z_dim, hidden_layer, output_dim, sttdev=0.001)

    def get_hidden_dim(self):
        return self._hidden_layer

    def get_weight(self, meta_v, no_grad=True):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1 = self.input_layer(z)
                w2, b2 = self.hidden(z)
                w3, b3 = self.last_layer(z)
        else:
            z = self.hyper(meta_v)
            w1, b1 = self.input_layer(z)
            w2, b2 = self.hidden(z)
            w3, b3 = self.last_layer(z)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)]

    def forward(self, meta_v, base_v):
        # produce dynmaic weights
        z = self.hyper(meta_v)
        w1, b1 = self.input_layer(z)
        w2, b2 = self.hidden(z)
        w3, b3 = self.last_layer(z)
        # dynamic network pass
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, out) + b2)
        out = torch.bmm(w3, out) + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v):
        with torch.no_grad():
            # produce dynmaic weights
            z = self.hyper(meta_v)
            w1, b1 = self.input_layer(z)
            w2, b2 = self.hidden(z)
            w3, b3 = self.last_layer(z)
            # dynamic network pass
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetworkCH(nn.Module):
    # meta_v --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, base_v_dim, output_dim, hidden_layer=256, hyper_emb_dim=1024):
        super(HyperNetworkCH, self).__init__()
        z_dim = hyper_emb_dim
        self._hidden_layer = hidden_layer
        # encode the meta_v into a hidden embedding
        self.hyper = Meta_Embadding(meta_v_dim, z_dim)
        # Q function net, use the hidden embedding to generate weights
        self.input_layer = Head(z_dim, base_v_dim + meta_v_dim, hidden_layer, sttdev=0.05)
        self.hidden = Head(z_dim, hidden_layer + meta_v_dim, hidden_layer, sttdev=0.008)
        self.last_layer = Head(z_dim, hidden_layer + meta_v_dim, output_dim, sttdev=0.001)

    def get_hidden_dim(self):
        return self._hidden_layer

    def get_weight(self, meta_v, no_grad=True):
        if no_grad:
            with torch.no_grad():
                z = self.hyper(meta_v)
                w1, b1 = self.input_layer(z)
                w2, b2 = self.hidden(z)
                w3, b3 = self.last_layer(z)
        else:
            z = self.hyper(meta_v)
            w1, b1 = self.input_layer(z)
            w2, b2 = self.hidden(z)
            w3, b3 = self.last_layer(z)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)]

    def forward(self, meta_v, base_v):
        # produce dynmaic weights
        z = self.hyper(meta_v)
        w1, b1 = self.input_layer(z)
        w2, b2 = self.hidden(z)
        w3, b3 = self.last_layer(z)
        # dynamic network pass
        out = F.relu(torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, torch.cat([out, meta_v.unsqueeze(2)], dim=1)) + b2)
        out = torch.bmm(w3, torch.cat([out, meta_v.unsqueeze(2)], dim=1)) + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v):
        with torch.no_grad():
            # produce dynmaic weights
            z = self.hyper(meta_v)
            w1, b1 = self.input_layer(z)
            w2, b2 = self.hidden(z)
            w3, b3 = self.last_layer(z)
            # dynamic network pass
            out_0 = torch.bmm(w1, torch.cat([base_v, meta_v], dim=1).unsqueeze(2)) + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, torch.cat([out_1, meta_v.unsqueeze(2)], dim=1)) + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, torch.cat([out_3, meta_v.unsqueeze(2)], dim=1)) + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetworkEmbedInput(nn.Module):
    # meta_v --> embedding --> hypernet, input layer of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hidden_layer=256, hyper_emb_dim=1024, clip_emb=False):
        super(HyperNetworkEmbedInput, self).__init__()
        z_dim = hyper_emb_dim
        self.clip_emb = clip_emb
        self._hidden_layer = hidden_layer
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        # encode the meta_v into a hidden embedding
        self.hyper = Meta_Embadding(meta_v_emb_dim, z_dim)
        # Q function net, use the hidden embedding to generate weights
        self.input_layer = Head(z_dim, base_v_dim + meta_v_emb_dim, hidden_layer, sttdev=0.05)
        self.hidden = Head(z_dim, hidden_layer, hidden_layer, sttdev=0.008)
        self.last_layer = Head(z_dim, hidden_layer, output_dim, sttdev=0.001)

    def get_hidden_dim(self):
        return self._hidden_layer

    def get_weight(self, meta_v, no_grad=True):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                z = self.hyper(emb)
                w1, b1 = self.input_layer(z)
                w2, b2 = self.hidden(z)
                w3, b3 = self.last_layer(z)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            z = self.hyper(emb.clone().detach())
            w1, b1 = self.input_layer(z)
            w2, b2 = self.hidden(z)
            w3, b3 = self.last_layer(z)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None):
        # produce dynmaic weights
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = F.relu(emb)
                if pos_emb is not None:
                    emb += pos_emb
            z = self.hyper(emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = F.relu(emb)
            if pos_emb is not None:
                emb += pos_emb
            z = self.hyper(emb.clone().detach())
        w1, b1 = self.input_layer(z)
        w2, b2 = self.hidden(z)
        w3, b3 = self.last_layer(z)
        # dynamic network pass
        out = F.relu(torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, out) + b2)
        out = torch.bmm(w3, out) + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = F.relu(emb)
            if pos_emb is not None:
                emb += pos_emb
            # produce dynmaic weights
            z = self.hyper(meta_v)
            w1, b1 = self.input_layer(z)
            w2, b2 = self.hidden(z)
            w3, b3 = self.last_layer(z)
            # dynamic network pass
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, out_1) + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, out_3) + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]


class HyperNetworkEmbedInputCH(nn.Module):
    # meta_v --> embedding --> hypernet, all layers of dynamicnet
    def __init__(self, meta_v_dim, meta_v_emb_dim, base_v_dim, output_dim, hidden_layer=256, hyper_emb_dim=1024, clip_emb=False):
        super(HyperNetworkEmbedInputCH, self).__init__()
        z_dim = hyper_emb_dim
        self._hidden_layer = hidden_layer
        self.clip_emb = clip_emb
        self.meta_v_emb_layer = nn.Linear(meta_v_dim, meta_v_emb_dim)
        self.meta_v_emb_layer.apply(init_weights_kaiming_uniform)
        # encode the meta_v into a hidden embedding
        self.hyper = Meta_Embadding(meta_v_emb_dim, z_dim)
        # Q function net, use the hidden embedding to generate weights
        self.input_layer = Head(z_dim, base_v_dim + meta_v_emb_dim, hidden_layer, sttdev=0.05)
        self.hidden = Head(z_dim, hidden_layer + meta_v_emb_dim, hidden_layer, sttdev=0.008)
        self.last_layer = Head(z_dim, hidden_layer + meta_v_emb_dim, output_dim, sttdev=0.001)

    def get_hidden_dim(self):
        return self._hidden_layer

    def get_weight(self, meta_v, no_grad=True):
        if no_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                z = self.hyper(emb)
                w1, b1 = self.input_layer(z)
                w2, b2 = self.hidden(z)
                w3, b3 = self.last_layer(z)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            z = self.hyper(emb.clone().detach())
            w1, b1 = self.input_layer(z)
            w2, b2 = self.hidden(z)
            w3, b3 = self.last_layer(z)
        return [torch.flatten(w1, start_dim=1), torch.flatten(w2, start_dim=1), torch.flatten(w3, start_dim=1)], \
               [torch.flatten(b1, start_dim=1), torch.flatten(b2, start_dim=1), torch.flatten(b3, start_dim=1)]

    def forward(self, meta_v, base_v, stop_meta_v_emb_grad=False, pos_emb=None):
        # produce dynmaic weights
        if stop_meta_v_emb_grad:
            with torch.no_grad():
                emb = self.meta_v_emb_layer(meta_v)
                if self.clip_emb:
                    emb = F.relu(emb)
                if pos_emb is not None:
                    emb += pos_emb
            z = self.hyper(emb)
        else:
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = F.relu(emb)
            if pos_emb is not None:
                emb += pos_emb
            z = self.hyper(emb.clone().detach())
        w1, b1 = self.input_layer(z)
        w2, b2 = self.hidden(z)
        w3, b3 = self.last_layer(z)
        # dynamic network pass
        out = F.relu(torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) + b1)
        out = F.relu(torch.bmm(w2, torch.cat([out, emb.clone().detach().unsqueeze(2)], dim=1)) + b2)
        out = torch.bmm(w3, torch.cat([out, emb.clone().detach().unsqueeze(2)], dim=1)) + b3
        out = torch.squeeze(out, dim=2)
        return out

    def get_layer_out(self, meta_v, base_v, pos_emb=None):
        with torch.no_grad():
            emb = self.meta_v_emb_layer(meta_v)
            if self.clip_emb:
                emb = F.relu(emb)
            if pos_emb is not None:
                emb += pos_emb
            # produce dynmaic weights
            z = self.hyper(meta_v)
            w1, b1 = self.input_layer(z)
            w2, b2 = self.hidden(z)
            w3, b3 = self.last_layer(z)
            # dynamic network pass
            out_0 = torch.bmm(w1, torch.cat([base_v, emb], dim=1).unsqueeze(2)) + b1
            out_1 = F.relu(out_0)
            out_2 = torch.bmm(w2, torch.cat([out_1, emb.unsqueeze(2)], dim=1)) + b2
            out_3 = F.relu(out_2)
            out_4 = torch.bmm(w3, torch.cat([out_3, emb.unsqueeze(2)], dim=1)) + b3
            out_4 = torch.squeeze(out_4, dim=2)
        return [out_1, out_3, out_4]