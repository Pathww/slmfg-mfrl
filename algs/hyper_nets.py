import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.s1 = nn.Linear(h_layer, output_dim_out)

        self.init_layers(sttdev)

    def forward(self, x):
        # weights, bias and scale for dynamic layer
        w = self.W1(x).view(-1, self.output_dim_out, self.output_dim_in)
        b = self.b1(x).view(-1, self.output_dim_out, 1)
        s = 1.0 + self.s1(x).view(-1, self.output_dim_out, 1)

        return w, b, s

    def init_layers(self, stddev):
        torch.nn.init.uniform_(self.W1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.b1.weight, -stddev, stddev)
        torch.nn.init.uniform_(self.s1.weight, -stddev, stddev)

        torch.nn.init.zeros_(self.W1.bias)
        torch.nn.init.zeros_(self.b1.bias)
        torch.nn.init.zeros_(self.s1.bias)


class Meta_Embadding(nn.Module):
    def __init__(self, meta_dim, z_dim):

        super(Meta_Embadding, self).__init__()

        self.z_dim = z_dim

        self.hyper = nn.Sequential(
            nn.Linear(meta_dim, 256),
            ResBlock(256, 256),
            # ResBlock(256, 256),
            nn.Linear(256, 512),
            ResBlock(512, 512),
            # ResBlock(512, 512),
            nn.Linear(512, 1024),
            ResBlock(1024, 1024),
            # ResBlock(1024, 1024),
        )

        self.init_layers()

    def forward(self, meta_v):
        z = self.hyper(meta_v).view(-1, self.z_dim)
        return z

    def init_layers(self):

        for module in self.hyper.modules():
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(
                    module.weight
                )
                bound = 1.0 / math.sqrt(2 * fan_in)
                torch.nn.init.uniform_(module.weight, -bound, bound)


class HyperNetwork(nn.Module):
    # Hyper net create weights with respect to the state and estimates function Q_s(a)
    def __init__(
        self, meta_v_dim, base_v_dim, output_dim, hidden_layer=256, z_dim=1024
    ):
        super(HyperNetwork, self).__init__()

        self._hidden_layer = hidden_layer
        # encode the meta_v into a hidden embedding
        self.hyper = Meta_Embadding(meta_v_dim, z_dim)

        # Q function net
        # use the hidden embedding to generate weights
        self.input_layer = Head(z_dim, base_v_dim, hidden_layer, sttdev=0.05)
        self.hidden = Head(z_dim, hidden_layer, hidden_layer, sttdev=0.008)
        self.last_layer = Head(z_dim, hidden_layer, output_dim, sttdev=0.001)

    def forward(self, meta_v, base_v):
        # produce dynmaic weights
        z = self.hyper(meta_v)
        w1, b1, s1 = self.input_layer(z)
        w2, b2, s2 = self.hidden(z)
        w3, b3, s3 = self.last_layer(z)

        # dynamic network pass
        out = F.relu(torch.bmm(w1, base_v.unsqueeze(2)) * s1 + b1)
        out = F.relu(torch.bmm(w2, out) * s2 + b2)
        out = torch.bmm(w3, out) * s3 + b3
        out = torch.squeeze(out, dim=2)

        return out
