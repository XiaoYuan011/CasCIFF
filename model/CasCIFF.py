import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
from .MyAE import AE
from .layers import MPL_Net, BiGCN
from .GlobalEmbing_AE_Raw import GlobalEmbing


class CasCIFF(nn.Module):
    def __init__(self, increase_node, feat_in, squence_length, input_dim, n_time_interval, emb_dim, z_dim):
        super(CasCIFF, self).__init__()
        self.squence_length = squence_length
        self.input_dim = input_dim
        self.n_time_interval = n_time_interval
        self.emb_dim = emb_dim
        self.z_dim = z_dim
        self.dens_hiddensize1 = 128
        self.dens_hiddensize2 = 64
        self.feat_in = feat_in
        self.increase_node = increase_node
        self.feat_hidden = 50
        self.feat_out = 64
        self.rnn_hidden = 128
        self.dropout = 0
        self.num_layers = 2
        self.n_steps = 100
        self.pad_node = self.n_steps % self.increase_node
        self.time_weight = torch.nn.Parameter(torch.Tensor(self.n_time_interval), requires_grad=True)
        self._set_layers()
        self._reset_params()

    def _reset_params(self):
        for p in self.parameters():
            if p.requires_grad:
                if len(p.shape) > 1:
                    init.kaiming_uniform_(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    init.uniform_(p, a=-stdv, b=stdv)

    def MeanSquaredLogarithmicError(self, pred, truth):
        first_log = torch.log2(torch.clamp(pred, 1e-7, None) + 1.)
        second_log = torch.log2(torch.clamp(truth, 1e-7, None) + 1.)
        return torch.nn.functional.mse_loss(first_log, second_log)

    def _set_layers(self):
        self.emb_layer = GlobalEmbing(squence_length=self.squence_length, input_dim=self.input_dim,
                                      emb_dim=self.emb_dim, z_dim=self.z_dim)

        self.bigcn = BiGCN(input_features=self.feat_in, hidden_features=self.feat_hidden, output_features=self.feat_out)
        self.ae = AE(input_dim=self.feat_out + self.z_dim, h_dim=96, z_dim=self.z_dim)
        self.biGRU = nn.GRU(input_size=self.z_dim + 1, hidden_size=self.rnn_hidden, bidirectional=True,
                            batch_first=True, num_layers=self.num_layers, dropout=self.dropout)
        self.mlp2 = MPL_Net(dens_inputsize=self.rnn_hidden * 2, dens_outputsize=1, dens_dropout=self.dropout,
                           dens_hiddensize1=self.dens_hiddensize1, dens_hiddensize2=self.dens_hiddensize2)

    def forward(self, batch_x, batch_l, batch_global_x, batch_y, batch_label, batch_time_interval, batch_rnn_index,
                batch_time_serise):

        total_loss, total_classfier_loss, total_classfier_acc, batch_z = self.emb_layer(batch_global_x, batch_label)
        if self.increase_node > 1:
            batch_temp_z = torch.reshape(batch_z[:, :(self.n_steps // self.increase_node) * self.increase_node],
                                         [batch_z.shape[0], -1, self.increase_node, batch_z.shape[2]])
            batch_temp_z = torch.sum(batch_temp_z, dim=2, keepdim=False)
            if self.pad_node == 1:
                batch_z_padding = batch_z[:, -1]
                batch_z = torch.cat([batch_temp_z, batch_z_padding.unsqueeze(1)], dim=1)
            elif self.pad_node > 1:
                batch_z_padding = torch.sum(batch_z[:, -self.pad_node:], dim=1, keepdim=False)
                batch_z = torch.cat([batch_temp_z, batch_z_padding.unsqueeze(1)], dim=1)
            else:
                batch_z = batch_temp_z
        batch_repretation = self.bigcn(batch_x, batch_l)
        batch_repretation = torch.mean(batch_repretation, dim=2)
        batch_repretation = torch.cat([batch_repretation, batch_z], dim=-1)
        vae_loss, batch_z_repretation = self.ae(batch_repretation)

        if self.increase_node > 1:
            batch_time_serise_subgs = batch_time_serise.squeeze(1)[:, self.increase_node - 1::self.increase_node]
            if self.pad_node != 0:
                batch_time_serise_subgs = torch.cat([batch_time_serise_subgs,
                                                     batch_time_serise.squeeze(1)[:, -1].unsqueeze(-1)], dim=-1)
            batch_z_repretation = torch.cat([batch_z_repretation, batch_time_serise_subgs.unsqueeze(-1)], dim=-1)
        else:
            batch_z_repretation = torch.cat([batch_z_repretation, batch_time_serise.squeeze(1).unsqueeze(-1)], dim=-1)
        rnn_repretation, _ = self.biGRU(batch_z_repretation)

        time_decay = torch.matmul(batch_time_interval, self.time_weight.unsqueeze(-1))
        rnn_repretation = rnn_repretation * time_decay

        batch_rnn_index = batch_rnn_index.unsqueeze(-1).transpose(1, 2)
        graph_representation = torch.bmm(batch_rnn_index, rnn_repretation).squeeze(1)

        pred = self.mlp2(graph_representation)
        regression_loss = self.MeanSquaredLogarithmicError(pred, batch_y)
        return total_loss + regression_loss + vae_loss, total_classfier_loss, regression_loss, total_classfier_acc, pred
