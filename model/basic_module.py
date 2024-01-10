import torch
import torch.nn as nn
import math
from torch.nn import functional as F


class meta_weight_and_initial(nn.Module):
    def __init__(self, meta_dim, input_dim, output_dim, bias=True):
        super(meta_weight_and_initial, self).__init__()
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.generate_weight = nn.Linear(meta_dim, input_dim * output_dim)
        if self.bias:
            self.generate_bias = nn.Linear(meta_dim, 1 * output_dim)

    def forward(self, mk):
        '''
        :param mk: B, N, C
        :param x: B, N, C
        :return:
        '''
        B, N, C = mk.shape
        weight = self.generate_weight(mk).reshape(B, N, self.input_dim, self.output_dim)
        if self.bias:
            bias = self.generate_bias(mk).reshape(B, 1, N, self.output_dim)
            return weight, bias
        return weight




class weight_and_initial(nn.Module):
    def __init__(self, input_dim, output_dim, num=1, bias=True):
        super(weight_and_initial, self).__init__()

        if num == 1:
            self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
            if bias:
                self.bias = nn.Parameter(torch.empty(1, output_dim))
            else:
                self.bias = None

        else:
            self.weight = nn.Parameter(torch.empty(num, input_dim, output_dim))
            if bias:
                self.bias = nn.Parameter(torch.empty(num, 1, output_dim))
            else:
                self.bias = None

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self):
        if self.bias == None:
            return self.weight
        else:
            return self.weight, self.bias


class meta_Linear(nn.Module):
    def __init__(self, meta_dim, input_dim, output_dim, bias=True):
        super(meta_Linear, self).__init__()
        self.bias = bias
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.get_meta_weight = meta_weight_and_initial(meta_dim, input_dim, output_dim)  # weight: C, C*D, bias: C, D

    def forward(self, mk, x):
        '''
        :param mk: B, N, C
        :param x: B, N, C
        :return:
        '''
        weight, bias = self.get_meta_weight(mk)
        if self.bias:
            return torch.einsum('btnc, bncd -> btnd', x, weight) + bias
        return torch.einsum('btnc, bncd -> btnd', x, weight)


class glu(nn.Module):
    def __init__(self, d_model, receptive_length=1, type="linear", bias=True):
        super(glu, self).__init__()
        self.d_model = d_model
        self.type = type
        if type == "linear":
            self.fc = nn.Linear(d_model, 2 * d_model, bias)
        if type == "conv":
            self.conv = nn.Conv2d(d_model, 2 * d_model, (1, receptive_length), stride=(1, 1))

    def forward(self, x):
        # (B, T, N, C) @ (C, 2C) = B, T, N, 2C
        if self.type == "linear":
            x = self.fc(x)
        if self.type == "conv":
            x = x.permute(0, 3, 2, 1)
            x = self.conv(x)
            x = x.permute(0, 3, 2, 1)

        # split(B, T, N, 2C) = (B, T, N, C), (B, T, N, C)
        lhs, rhs = torch.split(x, self.d_model, dim=-1)
        return lhs * torch.sigmoid(rhs)


class meta_glu(nn.Module):
    def __init__(self, meta_dim, d_model, type="linear"):
        super(meta_glu, self).__init__()
        self.d_model = d_model
        self.type = type
        if type == "linear":
            self.fc = meta_Linear(meta_dim, d_model, 2 * d_model)


    def forward(self, mk, x):
        # (B, T, N, C) @ (C, 2C) = B, T, N, 2C
        if self.type == "linear":
            x = self.fc(mk, x)
        # split(B, T, N, 2C) = (B, T, N, C), (B, T, N, C)
        lhs, rhs = torch.split(x, self.d_model, dim=-1)
        return lhs * torch.sigmoid(rhs)


class matrix_decomposition(nn.Module):
    def __init__(self, d1, d2, r, num=1):
        super(matrix_decomposition, self).__init__()
        self.emb1 = weight_and_initial(d1, r, num, bias=None)
        self.emb2 = weight_and_initial(r, d2, bias=None)

    def forward(self):

        return torch.matmul(self.emb1(), self.emb2())  # num, d1, d2





class generate_spatial_graph(nn.Module):
    def __init__(self, d_model, r):
        super(generate_spatial_graph, self).__init__()
        self.emb = nn.Linear(d_model, 2*r, bias=None)

    def forward(self, mk):
        emb1, emb2 = torch.chunk(self.emb(mk), 2, dim=-1)
        norm_emb1_2 = torch.norm(emb1, dim=-1, p=2).unsqueeze(-1)
        norm_emb2_2 = torch.norm(emb2, dim=-1, p=2).unsqueeze(-1)
        return torch.einsum('bnr, bmr -> bnm', emb1/norm_emb1_2, emb2/norm_emb2_2)


class generate_temporal_graph(nn.Module):
    def __init__(self, d_model, t):
        super(generate_temporal_graph, self).__init__()
        self.emb = nn.Linear(d_model, 2 * t, bias=None)

    def forward(self, mk):
        emb1, emb2 = torch.chunk(self.emb(mk).transpose(1, 2), 2, dim=1)
        norm_emb1_2 = torch.norm(emb1, dim=-1, p=2).unsqueeze(-1)
        norm_emb2_2 = torch.norm(emb2, dim=-1, p=2).unsqueeze(-1)

        return torch.einsum('bpn, bqn -> bpq', emb1 / norm_emb1_2, emb2 / norm_emb2_2)


def CMD(x1, x2, moments=2, element_wise=False):
    if element_wise:
        mu1, mu2 = torch.mean(x1, dim=(2, 3)), torch.mean(x2, dim=(2, 3))  # B, T // B, T
        cmds = torch.norm(mu1 - mu2, p=2, dim=-1)  # B
        x1, x2 = x1 - mu1.unsqueeze(-1).unsqueeze(-1), x2 - mu2.unsqueeze(-1).unsqueeze(-1)
        for i in range(2, moments + 1):
            t1 = torch.mean(x1 ** i, dim=(2, 3))  # B, T, N
            t2 = torch.mean(x2 ** i, dim=(2, 3))
            cmd = torch.norm(t1 - t2, p=2, dim=-1)  # B, K
            cmds = cmds + cmd
        return cmds  # B, K; B

    else:
        mu1, mu2 = torch.mean(x1, dim=(2, 3)), torch.mean(x2, dim=(2, 3))  # B, T // K, T
        cmds = torch.norm(mu1.unsqueeze(1) - mu2.unsqueeze(0), p=2, dim=2)  # B, K
        x1, x2 = x1 - mu1.unsqueeze(-1).unsqueeze(-1), x2 - mu2.unsqueeze(-1).unsqueeze(-1)  # B, T, N, C // B, T, N, C
        for i in range(2, moments + 1):
            t1 = torch.mean(x1 ** i, dim=(2, 3))  # B, T, N
            t2 = torch.mean(x2 ** i, dim=(2, 3))
            cmd = torch.norm(t1.unsqueeze(1) - t2.unsqueeze(0), p=2, dim=2)  # B, K
            cmds = cmds + cmd
        value, index = torch.min(cmds, dim=-1)
        return cmds, value, index  # B, K; B

