import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from .MyAE import AE as VAE
from .layers import MPL_Net
import numpy as np


class GlobalEmbing(nn.Module):
    def __init__(self, squence_length, input_dim, emb_dim, z_dim):
        super(GlobalEmbing, self).__init__()
        self.squence_length = squence_length
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.label_class = 2
        self.z_dim = z_dim
        self.dens_hiddensize1 = 32
        self.dens_hiddensize2 = 16
        self.dropout = 0
        self.n_steps = 100
        self.squence_weight = torch.nn.Parameter(torch.Tensor(self.squence_length), requires_grad=True)
        self._set_layers()
        # self._reset_params()

    def _reset_params(self):
        for p in self.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    init.xavier_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    init.uniform_(p, a=-stdv, b=stdv)

    def padding_z(self, z, global_x):
        out_z = []
        for i in range(global_x.shape[0]):
            temp_z = z[:global_x[i]]
            out_z.append(torch.cat([temp_z, torch.zeros(self.n_steps - temp_z.shape[0], self.z_dim, dtype=torch.float32,
                                                        requires_grad=False).cuda()], dim=0))
        out_z = torch.stack(out_z)
        return out_z

    def _set_layers(self):
        self.vae = VAE(h_dim=self.emb_dim, input_dim=self.input_dim, z_dim=self.z_dim)
        self.mlp = MPL_Net(dens_inputsize=self.z_dim, dens_outputsize=self.label_class, dens_dropout=self.dropout,
                           dens_hiddensize1=self.dens_hiddensize1, dens_hiddensize2=self.dens_hiddensize2)
        self.criterion = nn.CrossEntropyLoss(
            weight=torch.from_numpy(np.array([0.5 / 0.7, 0.5 / 0.3], dtype=np.float32)), reduction='mean')

    def forward(self, batch_global_x, batch_label):

        total_classfier_loss = []
        total_vae_loss = []
        total_classfier_acc = []
        batch_z = []
        for i in range(len(batch_global_x)):
            temp_input = batch_global_x[i]
            temp_label = batch_label[i]
            x = temp_input
            x = torch.transpose(x, 1, 2)
            x = torch.transpose(x * self.squence_weight, 1, 2)
            x = x.flatten(1)
            vae_loss, z = self.vae(x)
            pred1 = F.softmax(self.mlp(z), dim=-1)
            accuarcy = torch.eq(pred1.max(1)[1], temp_label).float().mean()
            classfier_loss = self.criterion(pred1, temp_label)
            total_classfier_loss.append(classfier_loss)
            total_vae_loss.append(vae_loss)
            total_classfier_acc.append(accuarcy)
            batch_z.append(torch.cat([z, torch.zeros(self.n_steps - z.shape[0], self.z_dim,
                                                     dtype=torch.float32, requires_grad=False).cuda()], dim=0))
        batch_z = torch.stack(batch_z)
        total_classfier_loss = torch.mean(torch.stack(total_classfier_loss))
        total_classfier_acc = torch.mean(torch.stack(total_classfier_acc))
        total_vae_loss = torch.mean(torch.stack(total_vae_loss))

        return total_vae_loss + total_classfier_loss, total_classfier_loss, total_classfier_acc, batch_z

