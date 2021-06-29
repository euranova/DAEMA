""" Contains the helper functions and constants. """

import os
import logging

import numpy as np
import torch
from torch import nn


def _create_and_return(path):
    """ Creates the folder if it does not exist and return the path.

    :param path: String; path of the folder to create
    :return: String; path of the created folder"""
    if not os.path.exists(path):
        logging.warning("creating %s", path)
        os.makedirs(path)
    return path


ROOT_PATH = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
DATA_PATH = _create_and_return(os.path.join(ROOT_PATH, "files", "data"))
RESULTS_PATH = _create_and_return(os.path.join(ROOT_PATH, "files", "results"))


def fix_seed(seed):
    """ Fixes the seeds of numpy and torch

    :param seed: Integer; seed to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


class ParallelLinear(nn.Module):
    """ Layer composed of parallel fully-connected layers.

    :param in_channels: Integer; number of input of each layer
    :param out_channels: Integer; number of output of each layer
    :param n_layers: Integer; number of parallel layers
    """
    def __init__(self, in_channels, out_channels, n_layers):
        super().__init__()
        self.dims = ((n_layers, in_channels), (n_layers, out_channels))
        self.layers = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(n_layers)
        ])

    def __repr__(self):
        return f'<ParallelLinear{self.dims}>'

    def forward(self, input_):
        # .T[:, i].T takes the dim just before the last one (no matter how many dimensions there are)
        out = [self.layers[i](input_.T[:, i].T) for i in range(self.dims[0][0])]
        return torch.stack(out, dim=len(input_.shape) - 2)


class View(nn.Module):
    """ Layer to reshape the data (keeping the first (batch) dimension as is).

    :param shape: tuple(Integer); expected shape (batch_dimension excluded)
    """
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'<View{self.shape}>'

    def forward(self, input_):
        return input_.view((input_.shape[0], *self.shape))
