from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import dropout_adj, to_dense_adj
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class EdgeDrop(nn.Module):
    def __init__(self, args):
        super(EdgeDrop, self).__init__()

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
        self.edge_dropout = args.edge_dropout

        self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=False))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=False))
            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=False))

    def forward(self, x, edge_index):

        # implemented based on GCNModel: https://github.com/DropEdge/DropEdge/blob/master/src/models.py
        # Baseblock MultiLayerGCNBlock with nbaseblocklayer=1

        edge_index, _ = dropout_adj(edge_index, p=self.edge_dropout,
                                    force_undirected=False, training=self.training)

        for i in range(self.num_layers-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            if self.type_norm == 'batch':
                x = self.layers_bn[i](x)
            x = F.relu(x)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index)
        return x

    def Dirichlet_energy(self, x, adj):
        # x = F.normalize(x, p=2, dim=1)
        x = torch.matmul(torch.matmul(x.t(), adj), x)
        energy = torch.trace(x)
        return energy.item()


    def compute_energy(self, x, edge_index, device):

        energy_list = []

        edge_weight = None
        edge_index_cached, edge_weight = gcn_norm(
            edge_index, edge_weight, x.size(0), False, dtype=x.dtype)
        adj_weight = to_dense_adj(edge_index_cached, edge_attr=edge_weight)
        num_nodes = x.size(0)
        adj_weight = torch.squeeze(adj_weight, dim=0)
        laplacian_weight = torch.eye(num_nodes, dtype=torch.float, device=device) - adj_weight

        # compute energy in the first layer
        energy = self.Dirichlet_energy(x, laplacian_weight)
        energy_list.append(energy)

        edge_index, _ = dropout_adj(edge_index, p=self.edge_dropout,
                                    force_undirected=False, training=self.training)

        for i in range(self.num_layers-1):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.layers_GCN[i](x, edge_index)
            if self.type_norm in ['batch', 'pair']:
                x = self.layers_bn[i](x)
            x = F.relu(x)

            # compute energy in the middle layer
            energy = self.Dirichlet_energy(x, laplacian_weight)
            energy_list.append(energy)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers_GCN[-1](x, edge_index)

        # compute energy in the last layer
        energy = self.Dirichlet_energy(x, laplacian_weight)
        energy_list.append(energy)
        return energy_list



