""" Contains the implementation of AimNet, from Holoclean."""

import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm


def _masked_loss_func(output, masks, samples):
    """ Computes the MSE loss.

    :param output: tensor(Float); imputed samples
    :param masks: tensor(Float); corresponding masks
    :param samples: tensor(Float); original samples (with missing values)
    :return: tensor(Float); loss obtained
    """
    if masks is None:
        masks = torch.zeros(samples.shape)
    mse_loss = torch.sum((~masks.bool() * samples - ~masks.bool() * output) ** 2)
    if torch.sum(1 - masks) > 0:
        mse_loss /= torch.sum(1 - masks)
    return mse_loss


class AimNet(nn.Module):
    """ AimNet architecture as introduced in the AimNet paper (for numerical features only).

    :param embedding_size: Integer: size of the embeddings
    :param n_cols: Integer: number of features
    :param dropout_percent: proportion of values to drop during training """
    def __init__(self, embedding_size, n_cols, dropout_percent=0.0):
        super().__init__()
        self.n_cols = n_cols
        self.num_module_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, embedding_size)
            )
            for _ in range(n_cols)
        ])
        self.q_module = nn.ModuleList([nn.Embedding(1, n_cols) for _ in range(n_cols)])
        self.continuous_target_projection = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embedding_size, embedding_size),
                nn.ReLU(),
                nn.Linear(embedding_size, 1)
            )
            for _ in range(n_cols)
        ])
        self.dropout_layer = nn.Dropout(dropout_percent)

    def forward(self, samples):
        """ Forward function

        :param samples: Tensor; samples with missing values
        :return: Tensor; imputed samples
        """

        v_matrix = torch.stack([
            layer(samples[:, i].view(-1, 1)) for i, layer in enumerate(self.num_module_list)], dim=1)
        v_matrix = self.dropout_layer(v_matrix)
        v_matrix = nn.functional.normalize(v_matrix, dim=2, p=2)
        q_matrix = torch.stack([layer(torch.tensor([0])) for layer in self.q_module], dim=0).squeeze()

        k_matrix = torch.eye(self.n_cols)
        preds = []
        for i in range(self.n_cols):
            weights = torch.nn.Softmax(dim=0)(torch.matmul(k_matrix[i], q_matrix))
            mask = torch.ones(self.n_cols)
            mask[i] = 0
            weights = torch.mul(weights, mask)

            context_vector = torch.matmul(weights, v_matrix)
            preds.append(self.continuous_target_projection[i](context_vector))

        return torch.cat(preds, dim=1)


class Holoclean:
    """ AimNet procedure as introduced in the AimNet paper (for numerical features only).

    :param samples: np.ndarray(Float); samples to use for initialisation
    :param masks: np.ndarray(Float); corresponding mask matrix
    :param args: ArgumentParser; arguments of the program (see pipeline/argument_parser.py)
    """
    def __init__(self, samples, masks, args):
        del masks  # Unused
        self.net = AimNet(args.holoclean_embedding_size, n_cols=samples.shape[1],
                          dropout_percent=args.holoclean_dropout)

    def train_generator(self, samples, masks, args):
        """ Trains the network epoch after epoch as a generator.

        :param samples: np.ndarray(Float); samples to use for training
        :param masks: np.ndarray(Float); corresponding mask matrix
        :param args: ArgumentParser; arguments of the program (see pipeline/argument_parser.py)
        :return: Integer; epoch number
        """
        if args.batch_size == 0:
            batch_size = 32 if len(samples) > 2000 else 1
        elif args.batch_size == -1:
            batch_size = len(samples)
        else:
            batch_size = args.batch_size

        self.net.train()

        opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 1, T_mult=1)
        dl = torch.utils.data.DataLoader(dataset=list(zip(samples, masks)), batch_size=batch_size, shuffle=True)
        iters = len(dl)
        step = 0
        yield step
        self.net.train()

        total_steps = max(args.metric_steps)
        for epoch in tqdm(range(total_steps)):
            for j, (batch_samples, batch_masks) in enumerate(dl):
                output = self.net(batch_samples)

                opt.zero_grad()
                loss = _masked_loss_func(output, batch_masks, batch_samples)
                loss.backward()
                opt.step()

                scheduler.step(epoch + (j / iters))
            step += 1
            yield step
            self.net.train()

    def test(self, samples, masks):
        """ Imputes the given samples using the network.

        :param samples: np.ndarray(Float); samples to impute
        :param masks: np.ndarray(Float); corresponding mask matrix
        :return: np.ndarray(Float); imputed samples
        """
        self.net.eval()
        imputed_data = self.net(torch.from_numpy(samples)).data.numpy()
        return imputed_data * masks + samples * (1-masks)
