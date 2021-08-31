""" Model implementing the MIDA paper, with some additional possibilities. """

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from .baseline_imputations import MeanImputation


def _init_weights(layer):
    """Initialises the weights of the layer.

    :param layer: nn.Module; layer to initialise the weights for
    """
    if isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight)
        layer.bias.data.fill_(0.01)


class DAE(nn.Module):
    """ DAE architecture used in the MIDA paper.

    :param n_cols: Integer: number of features
    :param theta: Integer: hyperparameter to control the width of the network (see paper)
    :param depth: Integer: hyperparameter to control the depth of the network (see paper)
    """
    def __init__(self, n_cols, theta=7, depth=3):
        super().__init__()

        encoder_modules = []
        decoder_modules = []
        for i in range(depth):
            encoder_modules.append(nn.Linear(n_cols + theta * i, n_cols + theta * (i + 1)))
            encoder_modules.append(nn.Tanh())
            decoder_modules.insert(0, nn.Tanh())
            decoder_modules.insert(0, nn.Linear(n_cols + theta * (i + 1), n_cols + theta * i))
        encoder_modules.pop()
        decoder_modules.pop()

        self.encoder = nn.Sequential(*encoder_modules)
        self.decoder = nn.Sequential(*decoder_modules)
        self.dropout = nn.Dropout(0.5)

    def forward(self, samples):
        """ Forward function

        :param samples: Tensor; samples with missing values
        :return: Tensor; imputed samples
        """
        samples = self.dropout(samples)
        samples = self.encoder(samples)
        samples = self.decoder(samples)
        return samples


class MIDA:
    """ MIDA procedure as introduced in the MIDA paper.

    :param samples: np.ndarray(Float); samples to use for initialisation
    :param masks: np.ndarray(Float); corresponding mask matrix
    :param args: ArgumentParser; arguments of the program (see pipeline/argument_parser.py)
    """

    def __init__(self, samples, masks, args):
        del masks  # Unused
        self.net = DAE(samples.shape[1], theta=args.mida_theta, depth=args.mida_depth)
        self.net.apply(_init_weights)

    def train_generator(self, samples, masks, args, **kwargs):
        """ Trains the network batch after batch as a generator.

        :param samples: np.ndarray(Float); samples to use for training
        :param masks: np.ndarray(Float); corresponding mask matrix
        :param args: ArgumentParser; arguments of the program (see pipeline/argument_parser.py)
        :param kwargs: keyword arguments to be passed to the Adam optimiser
        :return: Integer; step number
        """
        batch_size = samples.shape[0] if args.batch_size == -1 else args.batch_size
        mean_impute = MeanImputation(samples, masks, None)
        mean_impute.train(samples, masks, None)
        self.net.train()

        optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, **kwargs)
        criterion = nn.MSELoss()
        train_loader = torch.utils.data.DataLoader(
            dataset=list(zip(samples, masks)), batch_size=batch_size, shuffle=True)
        step = 0
        yield step
        self.net.train()
        total_steps = max(args.metric_steps)
        tqdm_ite = tqdm(range((total_steps // len(train_loader)) + 1))
        for _ in tqdm_ite:
            for batch_samples, batch_masks in train_loader:
                batch_samples = mean_impute.test(batch_samples, batch_masks)
                output = self.net(batch_samples)
                loss = criterion(output, batch_samples)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                step += 1
                yield step
                self.net.train()
                if step >= total_steps:
                    break
            if step >= total_steps:
                tqdm_ite.close()
                break

    def test(self, samples, masks):
        """ Imputes the given samples using the network.

        :param samples: np.ndarray(Float); samples to impute
        :param masks: np.ndarray(Float); corresponding mask matrix
        :return: np.ndarray(Float); imputed samples
        """
        self.net.eval()
        replace_missing = np.random.uniform(0, 0.01, samples.shape).astype(np.float32)
        samples = torch.from_numpy(samples * (1 - masks) + masks * replace_missing)
        masks = torch.from_numpy(masks)
        imputed_samples = self.net(samples)
        imputed_samples = samples * (1 - masks) + masks * imputed_samples
        return imputed_samples.data.numpy()
