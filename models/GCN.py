from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class pair_norm(torch.nn.Module):
    def __init__(self):
        super(pair_norm, self).__init__()

    def forward(self, x):
        col_mean = x.mean(dim=0)
        x = x - col_mean
        rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
        x = x / rownorm_mean
        return x


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.dropout = args.dropout
        self.cached =  args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.type_norm = args.type_norm

        self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached))

            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
            elif self.type_norm == 'pair':
                self.layers_bn.append(pair_norm())
        self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached))


    def forward(self, x, edge_index):

        # implemented based on DeepGCN: https://github.com/LingxiaoShawn/PairNorm/blob/master/models.py

        for i in range(self.num_layers-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            if self.type_norm in ['batch', 'pair']:
                x = self.layers_bn[i](x)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index)

        return x
