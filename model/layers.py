import torch
import torch.nn as nn
import torch.nn.init as init


class BiGCN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features):
        super(BiGCN, self).__init__()
        self.gcn1 = CheBy_conv(input_features, hidden_features)
        self.gcn2 = CheBy_conv(input_features + hidden_features, output_features)
        self.relu = nn.ReLU()

    def forward(self, x, L):
        x1 = self.relu(self.gcn1(x, L))
        x_ = torch.cat([x, x1], dim=-1)
        x_temp = self.relu(self.gcn2(x_, L))
        return x_temp


class MPL_Net(torch.nn.Module):
    def __init__(self, dens_hiddensize1, dens_hiddensize2, dens_inputsize, dens_outputsize, dens_dropout=0):
        super(MPL_Net, self).__init__()
        self.inputsize = dens_inputsize
        self.dens_hiddensize1 = dens_hiddensize1
        self.dens_hiddensize2 = dens_hiddensize2
        self.dens_dropout = dens_dropout
        self.outputsize = dens_outputsize
        self.setup_layers()

    def setup_layers(self):
        self.dens_net = torch.nn.Sequential(
            torch.nn.Linear(self.inputsize, self.dens_hiddensize1),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=self.dens_dropout),
            torch.nn.Linear(self.dens_hiddensize1, self.dens_hiddensize2),
            torch.nn.ReLU(True),
            torch.nn.Dropout(p=self.dens_dropout),
            torch.nn.Linear(self.dens_hiddensize2, self.outputsize)
        )

    def forward(self, x):
        return torch.relu(self.dens_net(x))


class CheBy_conv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(CheBy_conv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim), requires_grad=True)
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.weight)
        if self.use_bias:
            init.uniform_(self.bias)

    def forward(self, x, lap):

        support = torch.matmul(x, self.weight)
        output = torch.matmul(lap.unsqueeze(1), support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
