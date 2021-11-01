from torch import nn
import torch
import torch.nn.functional as F
from models.EGNN_layer import EGNNConv

class SReLU(nn.Module):
    """Shifted ReLU"""

    def __init__(self, nc, bias):
        super(SReLU, self).__init__()
        self.srelu_bias = nn.Parameter(torch.Tensor(nc,))
        self.srelu_relu = nn.ReLU(inplace=True)
        nn.init.constant_(self.srelu_bias, bias)

    def forward(self, x):
        return self.srelu_relu(x - self.srelu_bias) + self.srelu_bias

class EGNN(nn.Module):
    def __init__(self, args):
        super(EGNN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden

        self.cached = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.c_min = args.c_min
        self.c_max = args.c_max
        self.beta = args.beta

        self.bias_SReLU = args.bias_SReLU
        self.dropout = args.dropout
        self.output_dropout = args.output_dropout

        self.reg_params = []
        for i in range(self.num_layers):
            c_max = self.c_max if i==0 else 1.
            self.layers_GCN.append(EGNNConv(self.dim_hidden, self.dim_hidden,
                                           c_max=c_max, cached=self.cached, bias=False))
            self.layers_activation.append(SReLU(self.dim_hidden, self.bias_SReLU))
            self.reg_params.append(self.layers_GCN[-1].weight)
            if self.dataset == 'ogbn-arxiv':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))


        self.input_layer = torch.nn.Linear(self.num_feats, self.dim_hidden)
        self.output_layer = torch.nn.Linear(self.dim_hidden, self.num_classes)
        self.non_reg_params = list(self.input_layer.parameters()) + \
                              list(self.output_layer.parameters())
        self.srelu_params = list(self.layers_activation[:-1].parameters())

        if self.dataset in ['ogbn-arxiv']:
            self.bn = torch.nn.BatchNorm1d(self.dim_hidden)
            self.input_layer2 = torch.nn.Linear(self.dim_hidden, self.dim_hidden)
            self.non_reg_params += list(self.bn.parameters())
            self.non_reg_params += list(self.input_layer2.parameters())
            self.non_reg_params += list(self.layers_bn[:-1].parameters())



    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_layer(x)
        if self.dataset == 'ogbn-arxiv':
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.input_layer2(x)
        else:
            x = F.relu(x)

        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.beta
            x = self.layers_GCN[i](x, edge_index, original_x,
                                   beta=self.beta, residual_weight=residual_weight)

            if self.dataset == 'ogbn-arxiv':
                x = self.layers_bn[i](x)
            x = self.layers_activation[i](x)

        x = F.dropout(x, p=self.output_dropout, training=self.training)
        x = self.output_layer(x)

        return x








