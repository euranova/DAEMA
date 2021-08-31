""" Model implementing the DAEMA paper """

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

from .baseline_imputations import MeanImputation


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


class Generator(nn.Module):
    """ Architecture of the DAEMA model

    :param n_cols: Int; number of columns in the dataset
    :param mask_input: Generator.FC, Generator.ELEMENTWISE or None; what input to use for the feature encoder
        - Generator.FC: Uses masks concatenated to the corresponding samples as input of the feature encoder
        - Generator.ELEMENTWISE: Uses masks to impute the samples with learned values
        - None: Uses only the samples as input of the feature encoder
    :param feature_size: (Int or None, Int or None) or None; (d', d_z) from the paper ((ways, latent_dim))
    :param attention_mode: "classic", "full", "sep" or "no"; type of attention to use
        - full: as done in the paper, one set of weights per feature
        - classic: one set of weights for all features
        - sep: same as classic, but having d' independent networks to produce each latent vector version
        - no: no attention at all (classical denoising autoencoder)
    :param activation: Str or None; torch.nn activation function to use at the end of the network
        (or None for no activation)
    """

    ELEMENTWISE = 0
    FC = 1
    MODES = {FC: "_FC", ELEMENTWISE: "_EW", None: "_NO"}

    def __init__(self, n_cols, mask_input, feature_size, attention_mode, activation):
        super().__init__()

        assert attention_mode in ["classic", "full", "sep", "no"]
        if feature_size is None:
            feature_size = (None, None)
        if feature_size[0] is None:
            feature_size = (n_cols * 2, feature_size[1])
        if feature_size[1] is None:
            feature_size = (feature_size[0], n_cols * 2)
        activation_tup = (getattr(nn, activation)(),) if activation is not None else ()

        self.mask_input = mask_input
        self.feature_size = feature_size
        self.attention_mode = attention_mode

        self.pre_input = None
        encoder_in_dim = n_cols
        if self.mask_input == Generator.ELEMENTWISE:
            # initialise as if self.mask_input is None
            self.pre_input = nn.Parameter(torch.zeros((1, n_cols)), requires_grad=True)
        elif self.mask_input == Generator.FC:
            encoder_in_dim = 2 * n_cols

        if attention_mode == "no":
            self.features = nn.Sequential(
                nn.Linear(encoder_in_dim, np.prod(feature_size)),
                nn.Tanh(),
                nn.Linear(np.prod(feature_size), feature_size[1]),
                nn.Tanh()
            )
        else:
            self.features = nn.Sequential(
                nn.Linear(encoder_in_dim, np.prod(feature_size)),
                nn.Tanh(),
                View(feature_size),
                ParallelLinear(feature_size[1], feature_size[1], n_layers=feature_size[0]),
                nn.Tanh()
            ) if attention_mode == 'sep' else nn.Sequential(
                nn.Linear(encoder_in_dim, np.prod(feature_size)),
                nn.Tanh(),
                nn.Linear(np.prod(feature_size), np.prod(feature_size)),
                nn.Tanh(),
                View(feature_size)
            )

            self.attention = nn.Sequential(
                nn.Linear(n_cols, feature_size[0]),
                nn.Softmax(dim=1),
                View((1, feature_size[0]))
            ) if attention_mode != 'full' else nn.Sequential(
                nn.Linear(n_cols, np.prod(feature_size)),
                View(feature_size),
                nn.Softmax(dim=1)
            )

        self.output = nn.Sequential(
            nn.Linear(feature_size[-1], n_cols),
            *activation_tup
        )

    def forward(self, samples, masks):
        """ Forward function

        :param samples: Tensor; samples with missing values
        :param masks: Tensor; corresponding masks
        :return: Tensor; imputed samples
        """

        # ELEMENTWISE allow the network to choose the value for "missing"
        input_ = (samples + self.pre_input * masks if self.mask_input == Generator.ELEMENTWISE else
                  torch.cat((samples, masks), dim=1) if self.mask_input == Generator.FC else
                  samples)

        features = self.features(input_)
        if self.attention_mode != "no":
            attention = self.attention(masks)
            if self.attention_mode == 'full':
                features = (attention * features).sum(dim=1)
            else:
                features = attention.matmul(features).squeeze(dim=1)
        output = self.output(features)

        return output


class Daema:
    """ DAEMA model as presented in the paper.

    :param samples: np.ndarray(Float); samples to use for initialisation
    :param masks: np.ndarray(Float); corresponding mask matrix
    :param args: ArgumentParser; arguments of the program (see pipeline/argument_parser.py)
    """
    def __init__(self, samples, masks, args):
        del masks  # Unused
        mask_input = getattr(Generator, args.daema_mask_input) if args.daema_mask_input is not None else None
        feature_size = (args.daema_ways, args.daema_feats)
        self.net = Generator(samples.shape[1], mask_input, feature_size, args.daema_attention_mode,
                             args.daema_activation)

    def train_generator(self, samples, masks, args, **kwargs):
        """ Trains the network batch after batch as a generator.

        :param samples: np.ndarray(Float); samples to use for training
        :param masks: np.ndarray(Float); corresponding mask matrix
        :param args: ArgumentParser; arguments of the program (see pipeline/argument_parser.py)
        :param kwargs: keyword arguments to be passed to the Adam optimiser
        :return: Integer; step number
        """
        self.net.train()
        mean_impute = None
        if args.daema_loss_type == "full":
            mean_impute = MeanImputation(samples, masks, None)
            mean_impute.train(samples, masks, None)

        opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, **kwargs)
        dl = torch.utils.data.DataLoader(dataset=list(zip(samples, masks)), batch_size=args.batch_size, shuffle=True)

        step = 0
        yield step
        self.net.train()

        total_steps = max(args.metric_steps)
        tqdm_iter = tqdm(range((total_steps // len(dl)) + 1))
        for _ in tqdm_iter:
            for batch_samples, batch_masks in dl:
                keep = (np.random.uniform(0, 1, batch_samples.shape) > args.daema_pre_drop)
                new_masks = 1 - (1 - batch_masks) * keep
                output = self.net(batch_samples * keep, new_masks)

                loss = (
                    (1 - batch_masks) * ((output - batch_samples) ** 2) if args.daema_loss_type == "classic" else
                    (output - mean_impute.test(batch_samples, batch_masks)) ** 2 if args.daema_loss_type == "full" else
                    (1 - batch_masks) * new_masks * ((output - batch_samples) ** 2)
                ).sum(dim=0)

                opt.zero_grad()
                loss.sum().backward()
                opt.step()

                step += 1
                yield step
                self.net.train()
                if step >= total_steps:
                    break
            if step >= total_steps:
                tqdm_iter.close()
                break

    def test(self, samples, masks):
        """ Imputes the given samples using the network.

        :param samples: np.ndarray(Float); samples to impute
        :param masks: np.ndarray(Float); corresponding mask matrix
        :return: np.ndarray(Float); imputed samples
        """

        self.net.eval()
        t_samples, t_masks = torch.from_numpy(samples), torch.from_numpy(masks)
        t_output = self.net(t_samples, t_masks)
        output = t_output.data.numpy()
        return output * masks + samples * (1-masks)
