"""Multi-task implementation of Holoclean"""
import logging

import torch
import torch.utils.data
import torch.nn as nn
from tqdm import tqdm
import numpy as np


def masked_loss_func(output, mask, batch_data):
    if mask is None:
        mask = torch.zeros(batch_data.shape)
    MSE_loss = torch.sum((~mask.bool() * batch_data - ~mask.bool() * output) ** 2)
    if torch.sum(1 - mask) > 0:
        MSE_loss /= torch.sum(1 - mask)
    return MSE_loss


class AimNet(nn.Module):
    def __init__(self, k, n_cols, dropout_percent=0.0):
        super(AimNet, self).__init__()
        self.k = k
        self.n_cols = n_cols
        self.num_module_list = nn.ModuleList()
        self.continuous_target_projection = nn.ModuleList()
        self.Q_module = nn.ModuleList()
        for i in range(n_cols):
            seq_temp = nn.Sequential(nn.Linear(1, k),
                                     nn.ReLU(),
                                     nn.Linear(k, k)
                                     )
            self.num_module_list.append(seq_temp)
        # print(self.num_module_list)

        for i in range(n_cols):
            self.Q_module.append(nn.Embedding(1, n_cols))

        for i in range(n_cols):
            self.continuous_target_projection.append(nn.Sequential(
                nn.Linear(k, k), nn.ReLU(), nn.Linear(k, 1)))
        self.dropout_layer = nn.Dropout(dropout_percent)

    def forward(self, input_):
        V_list = []
        for i in range(self.n_cols):
            V_list.append(self.num_module_list[i](input_[:, i].view(-1, 1)))

        V_list = torch.stack(V_list, dim=1)
        V_list = self.dropout_layer(V_list)
        V_list = nn.functional.normalize(V_list, dim=2, p=2)
        Q_list = []
        for i in range(self.n_cols):
            temp_tensor = torch.tensor([0])
            Q_list.append(self.Q_module[i](temp_tensor))
        Q_list = torch.stack(Q_list, dim=0).squeeze()

        K = torch.eye(self.n_cols)
        preds = []
        for i in range(self.n_cols):
            temp = torch.nn.Softmax(dim=0)(torch.matmul(K[i], Q_list))
            mask = torch.ones(self.n_cols)
            mask[i] = 0
            temp = torch.mul(temp, mask)

            context_vector = torch.matmul(temp, V_list)
            preds.append(self.continuous_target_projection[i](context_vector))

        return torch.cat(preds, dim=1)


class Holoclean:
    def __init__(self, input_, mask, args):
        self.net = AimNet(args.holoclean_k, n_cols=input_.shape[1], dropout_percent=args.holoclean_dropout)

    def train_generator(self, input_, mask, args):
        if args.batch_size == 0:
            batch_size = 32 if len(input_) > 2000 else 1
        elif args.batch_size == -1:
            batch_size = len(input_)
        else:
            batch_size = args.batch_size

        self.net.train()

        opt = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, 1, T_mult=1)
        dl = torch.utils.data.DataLoader(dataset=list(zip(input_, mask)), batch_size=batch_size, shuffle=True)
        iters = len(dl)
        step = 0
        yield step
        self.net.train()

        total_steps = max(args.metric_steps)
        for epoch in tqdm(range(total_steps)):
            epoch_loss = 0
            for j, (batch_missing_data, batch_mask) in enumerate(dl):
                output = self.net(batch_missing_data)

                opt.zero_grad()

                loss = masked_loss_func(output, batch_mask, batch_missing_data)
                loss.backward()
                epoch_loss += loss.item()
                opt.step()

                scheduler.step(epoch + (j / iters))
            step += 1
            yield step
            self.net.train()

    def test(self, input_, mask):
        self.net.eval()
        imputed_data = self.net(torch.from_numpy(input_)).data.numpy()
        return imputed_data * mask + input_ * (1-mask)
