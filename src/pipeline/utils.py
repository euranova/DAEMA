import os
import logging

import numpy as np
import torch
from torch import nn


def write(s):
    s = str(s)
    with open("tmp.txt", "a") as f:
        f.write(" 0".join("\n".join(s.split(",")).split("-0")))


def _create_and_return(path):
    if not os.path.exists(path):
        logging.warning("creating %s", path)
        os.makedirs(path)
    return path


ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
DATA_PATH = _create_and_return(os.path.join(ROOT_PATH, "files", "data"))
RESULTS_PATH = _create_and_return(os.path.join(ROOT_PATH, "files", "results"))


def data_path(path):
    return os.path.join(DATA_PATH, path)


def fix_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


class ParallelLinear(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super().__init__()
        self.dims = ((n_layers, in_channels), (n_layers, out_channels))
        self.FC = nn.ModuleList()
        for _ in range(n_layers):
            self.FC.append(nn.Linear(in_channels, out_channels))

    def __repr__(self):
        return '<ParallelLinear{}>'.format(self.dims)

    def forward(self, input_):
        out = [self.FC[i](input_.T[:, i].T) for i in range(self.dims[0][0])]
        return torch.stack(out, dim=len(input_.shape) - 2)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return '<View{}>'.format(self.shape)

    def forward(self, input_):
        return input_.view((input_.shape[0], *self.shape))
