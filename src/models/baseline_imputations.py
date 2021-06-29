""" Basic imputation techniques. """

from collections import deque

import numpy as np


class ValueImputation:
    """ Puts the same chosen value everywhere."""

    def __init__(self, samples, masks, args, values=None):  # pylint:disable=unused-argument
        if values is None:
            values = 0
        self.values = values

    def train_generator(self, samples, masks, args):  # pylint:disable=no-self-use, unused-argument
        yield 0

    def test(self, samples, masks):
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values, dtype=np.float32)
        return (1 - masks) * samples + masks * self.values


class MeanImputation(ValueImputation):
    """ Performs mean imputation."""

    def train_generator(self, samples, masks, args):
        self.values = []
        for i in range(samples.shape[1]):
            if np.sum(1 - masks[:, i]) > 0:
                self.values.append(np.average(samples[:, i], weights=1 - masks[:, i], axis=0))
            else:
                self.values.append(0)
        yield 0

    def train(self, samples, masks, args):
        deque(self.train_generator(samples, masks, args), maxlen=0)


class Identity:
    """ Performs identity (no imputation). """
    def __init__(self, samples, masks, args):  # pylint:disable=no-self-use, unused-argument
        pass

    def train_generator(self, samples, masks, args):  # pylint:disable=no-self-use, unused-argument
        yield 0

    def test(self, samples, masks):  # pylint:disable=no-self-use, unused-argument
        return samples
