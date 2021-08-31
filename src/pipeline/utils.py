""" Contains the helper functions and constants. """

import os
import logging

import numpy as np
import torch


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


def fix_seed(seed):
    """ Fixes the seeds of numpy and torch

    :param seed: Integer; seed to use
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
