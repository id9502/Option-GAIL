import math
import torch
from torch import nn
from typing import Type


def init_layer(module, gain=math.sqrt(2)):
    with torch.no_grad():
        nn.init.orthogonal_(module.weight.data, gain=gain)
        nn.init.constant_(module.bias.data, 0)
    return module


def make_module(in_size, out_size, hidden, activation: Type[nn.Module] = nn.ReLU):
    n_in = in_size
    l_hidden = []
    for h in hidden:
        l_hidden.append(init_layer(torch.nn.Linear(n_in, h)))
        l_hidden.append(activation())
        n_in = h
    l_hidden.append(init_layer(torch.nn.Linear(n_in, out_size), gain=0.1))
    return torch.nn.Sequential(*l_hidden)


def make_module_list(in_size, out_size, hidden, n_net, activation: Type[nn.Module] = nn.ReLU):
    return nn.ModuleList([make_module(in_size, out_size, hidden, activation) for _ in range(n_net)])


def make_activation(act_name):
    return (torch.nn.ReLU if act_name == "relu" else
            torch.nn.Tanh if act_name == "tanh" else
            torch.nn.Sigmoid if act_name == "sigmoid" else
            torch.nn.Softplus if act_name == "softplus" else None)
