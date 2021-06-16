"""Model implementing the MIDA paper, with some additional possibilities."""

import logging

import torch
import torch.utils.data
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from models.baseline_imputations import MeanImputation


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)


class DAE(nn.Module):
    def __init__(self, n_cols, theta=7, depth=3):
        super(DAE, self).__init__()

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

    def forward(self, input_):
        input_ = self.dropout(input_)
        input_ = self.encoder(input_)
        input_ = self.decoder(input_)
        return input_


class MIDA:
    def __init__(self, input_, mask, args):
        """Initialises the model.

        :param input_: pd.DataFrame(Float); dataset to use for initialising
        :param mask: pd.DataFrame(Float); corresponding mask matrix
        :param args: ArgumentParser; arguments of the program
        """
        self.net = DAE(input_.shape[1], theta=args.mida_theta, depth=args.mida_depth)
        self.net.apply(init_weights)

    def train_generator(self, input_, mask, args, **kwargs):
        """Trains the Mida model.

        :param dataset: see pipeline_blocks.models.Model.train
        :param total_steps: Int; number of steps
        :param kwargs: arguments to be passed to the optimiser at initialisation (in addition to learning_rate)
        """
        batch_size = input_.shape[0] if args.batch_size == -1 else args.batch_size
        mean_impute = MeanImputation(input_, mask, None)
        mean_impute.train(input_, mask, None)
        self.net.train()

        optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr, **kwargs)
        criterion = nn.MSELoss()
        epoch_losses=[]
        train_loader = torch.utils.data.DataLoader(dataset=list(zip(input_, mask)), batch_size=batch_size, shuffle=True)
        iters = len(train_loader)
        i = epoch_loss = 0
        step = 0
        yield step
        self.net.train()
        total_steps = max(args.metric_steps)
        tqdm_ite = tqdm(range((total_steps // iters) + 1))
        early_stop = False
        for i in tqdm_ite:
            epoch_loss = 0
            for input_, mask in train_loader:
                input_ = mean_impute.test(input_, mask)
                reconst_data = self.net(input_)
                loss = criterion(reconst_data, input_)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                step += 1
                yield step
                self.net.train()
                if step >= total_steps or early_stop:
                    break
            if step >= total_steps or early_stop:
                tqdm_ite.close()
                break
            epoch_losses.append(epoch_loss)
            if i % max(step // 10, 1) == 0:
                logging.debug("epoch %d: MIDA loss: %f", i + 1, epoch_loss)
        logging.info("epoch %d: MIDA loss: %f", i + 1, epoch_loss)

    def test(self, input_, mask):
        self.net.eval()
        replace_missing = np.random.uniform(0, 0.01, input_.shape).astype(np.float32)
        input_ = torch.from_numpy(input_ * (1 - mask) + mask * replace_missing)
        mask = torch.from_numpy(mask)
        imputed_data = self.net(input_)
        imputed_data = input_ * (1 - mask) + mask * imputed_data
        return imputed_data.data.numpy()
