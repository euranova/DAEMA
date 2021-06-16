"""Model implementing the GAIN paper, with additional possibilities (e.g. MIDA as generator)."""

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np

from pipeline import utils
from models.baseline_imputations import MeanImputation


class Generator(nn.Module):
    ELEMENTWISE = 0
    FC = 1
    MODES = {FC: "_FC", ELEMENTWISE: "_EW", None: "_NO"}

    def __init__(self, n_cols, mask_input, feature_size, attention_mode, activation):
        """ Architecture of the DAEMA model

        :param n_cols: Int; number of columns in the dataset
        :param mask_input: Generator.FC, Generator.ELEMENTWISE or None; what input to use for the feature encoder
        :param feature_size: (Int or None, Int or None) or None; (d', d_z) from the paper ((ways, latent_dim))
        :param attention_mode: "classic", "full", "sep" or "no"; type of attention to use
        :param activation: Str or None; torch.nn activation function to use at the end of the network
            (or None for no activation)
        """
        super(Generator, self).__init__()

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
                utils.View(feature_size),
                utils.ParallelLinear(feature_size[1], feature_size[1], n_layers=feature_size[0]),
                nn.Tanh()
            ) if attention_mode == 'sep' else nn.Sequential(
                nn.Linear(encoder_in_dim, np.prod(feature_size)),
                nn.Tanh(),
                nn.Linear(np.prod(feature_size), np.prod(feature_size)),
                nn.Tanh(),
                utils.View(feature_size)
            )

            self.attention = nn.Sequential(
                nn.Linear(n_cols, feature_size[0]),
                nn.Softmax(dim=1),
                utils.View((1, feature_size[0]))
            ) if attention_mode != 'full' else nn.Sequential(
                nn.Linear(n_cols, np.prod(feature_size)),
                utils.View(feature_size),
                nn.Softmax(dim=1)
            )

        self.core = nn.Sequential()

        self.output = nn.Sequential(
            nn.Linear(feature_size[-1], n_cols),
            *activation_tup
        )

    def forward(self, data, mask):
        """

        :param data: Tensor; Data with missing values
        :param mask: Tensor; Mask matrix
        :return: (Tensor, Tensor); imputed inputs
        """

        # ELEMENTWISE allow the network to choose the value for "missing"
        input_ = (data + self.pre_input * torch.repeat_interleave(mask, torch.tensor(self.input_sizes), dim=1)
                  if self.mask_input == Generator.ELEMENTWISE else
                  torch.cat((data, mask), dim=1) if self.mask_input == Generator.FC else
                  data)

        features = self.features(input_)
        if self.attention_mode == 'no':
            core = self.core(features)
        else:
            attention = self.attention(mask)
            if self.attention_mode == 'full':
                core = self.core((attention * features).sum(dim=1))
            else:
                core = self.core(attention.matmul(features).squeeze(dim=1))
        output = self.output(core)

        return output


class Daema:
    def __init__(self, input_, mask, args):
        mask_input = getattr(Generator, args.daema_mask_input) if args.daema_mask_input is not None else None
        feature_size = (args.daema_ways, args.daema_feats)
        self.net = Generator(input_.shape[1], mask_input, feature_size, args.daema_attention_mode,
                             args.daema_activation)

    def train_generator(self, input_, mask, args, **kwargs):
        """ Trains the networks epoch after epoch as a generator.

        :param input_: pd.DataFrame(Float); dataset to use for training
        :param mask: pd.DataFrame(Float); corresponding mask matrix
        :param args: ArgumentParser; arguments of the program
        :param kwargs: Keyword args to be passed to torch.optim obj
        :return: Integer; step number
        """
        self.net.train()
        mean_impute = None
        if args.daema_loss_type == "full":
            mean_impute = MeanImputation(input_, mask, None)
            mean_impute.train(input_, mask, None)

        opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, **kwargs)
        dl = torch.utils.data.DataLoader(dataset=list(zip(input_, mask)), batch_size=args.batch_size, shuffle=True)

        step = 0
        yield step
        self.net.train()

        total_steps = max(args.metric_steps)
        tqdm_iter = tqdm(range((total_steps // len(dl)) + 1))
        for _ in tqdm_iter:
            for j, (input_, mask) in enumerate(dl):
                keep = (np.random.uniform(0, 1, input_.shape) > args.daema_pre_drop)
                new_mask = 1 - (1 - mask) * keep
                output = self.net(input_ * keep, new_mask)

                loss = (
                    ((1 - mask) * ((output - input_) ** 2)) if args.daema_loss_type == "classic" else
                    ((output - mean_impute.test(input_, mask)) ** 2) if args.daema_loss_type == "full" else
                    ((1 - mask) * new_mask * ((output - input_) ** 2))
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

    def test(self, input_, mask):
        self.net.eval()
        t_input_, t_mask = torch.from_numpy(input_), torch.from_numpy(mask)
        t_output = self.net(t_input_, t_mask)
        output = t_output.data.numpy()
        return output * mask + input_ * (1-mask)
