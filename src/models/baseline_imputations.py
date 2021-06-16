"""Baseline simple imputation techniques."""
from collections import deque

import numpy as np


class ValueImputation:
    """Put the same chosen value everywhere"""

    def __init__(self, input_, mask, args, values=None):
        if values is None:
            values = 0
        self.values = values

    def train_generator(self, input_, mask, args):  # pylint:disable=no-self-use  # for pipeline compatibility
        yield 0

    def test(self, input_, mask):
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values, dtype=np.float32)
        return (1 - mask) * input_ + mask * self.values


class MeanImputation(ValueImputation):
    """Classical mean imputation."""

    def train_generator(self, input_, mask, args):
        self.values = []
        for i in range(input_.shape[1]):
            if np.sum(1 - mask[:, i]) > 0:
                self.values.append(np.average(input_[:, i], weights=1 - mask[:, i], axis=0))
            else:
                self.values.append(0)
        yield 0

    def train(self, input_, mask, args):
        deque(self.train_generator(input_, mask, args), maxlen=0)


class Identity:
    def __init__(self, input_, mask, args):
        pass

    def train_generator(self, input_, mask, args):
        yield 0

    def test(self, input_, mask):
        return input_
