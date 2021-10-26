from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import numpy as np

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
        self.cached = self.transductive = args.transductive
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

        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))
        elif self.type_norm == 'pair':
            self.layers_bn.append(pair_norm())


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

        # weight = self.layers_GCN[-3].weight.cpu().detach().numpy()
        # print(weight)
        # print(np.max(weight), np.mean(weight), np.min(weight))

        return x

    def Dirichlet_energy(self, x, adj):
        # x = F.normalize(x, p=2, dim=1)
        x = torch.matmul(torch.matmul(x.t(), adj), x)
        energy = torch.trace(x)
        return energy.item()


    def compute_energy(self, x, edge_index, device):

        energy_list = []

        edge_index_cached, edge_weight = self.layers_GCN[0]._cached_edge_index
        adj_weight = to_dense_adj(edge_index_cached, edge_attr=edge_weight)
        num_nodes = x.size(0)
        adj_weight = torch.squeeze(adj_weight, dim=0)
        laplacian_weight = torch.eye(num_nodes, dtype=torch.float, device=device) - adj_weight

        # compute energy in the first layer
        energy = self.Dirichlet_energy(x, laplacian_weight)
        energy_list.append(energy)

        for i in range(self.num_layers-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            if self.type_norm in ['batch', 'pair']:
                x = self.layers_bn[i](x)
            x = F.relu(x)

            # compute energy in the middle layer
            energy = self.Dirichlet_energy(x, laplacian_weight)
            # in the higher layers of GCN, most of nodes will converge to zero vector
            # only some of the nodes have non-zero vectors, such abnormal nodes will leads
            # to very large value of Dirichlet energy
            # in this case, we abandon such abnornal nodes, and treat the Dirichlet energy as zero
            if i > 16 and energy > 1e-2:
                energy = 0.
            energy_list.append(energy)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index)




        # compute energy in the last layer
        energy = self.Dirichlet_energy(x, laplacian_weight)
        if self.num_layers > 16 and energy > 1e-2:
            energy = 0.
        energy_list.append(energy)
        return energy_list














