from torch import nn
import torch
import torch.nn.functional as F
from models.GCNII_layer import GCNIIdenseConv
import math
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj

class GCNII(nn.Module):
    def __init__(self, args):
        super(GCNII, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.dropout = args.dropout
        self.cached = self.transductive = args.transductive
        self.alpha = args.alpha
        self.lamda = args.lamda

        self.convs = torch.nn.ModuleList()
        self.layers_bn = nn.ModuleList()
        self.convs.append(torch.nn.Linear(self.num_feats, self.dim_hidden))
        for _ in range(self.num_layers):
            self.convs.append(GCNIIdenseConv(self.dim_hidden, self.dim_hidden))
            if self.dataset == 'ogbn-arxiv':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        self.convs.append(torch.nn.Linear(self.dim_hidden, self.num_classes))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters()) + list(self.convs[-1:].parameters())

        if self.dataset == 'ogbn-arxiv':
            self.non_reg_params += list(self.layers_bn[:-1].parameters())
            self.bn_input = torch.nn.BatchNorm1d(self.dim_hidden)
            self.non_reg_params += list(self.bn_input.parameters())


    def forward(self, x, edge_index):
        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[0](x)
        if self.dataset == 'ogbn-arxiv':
            x = self.bn_input(x)
        x = F.relu(x)
        _hidden.append(x)

        for i,con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout, training=self.training)
            beta = math.log(self.lamda/(i+1)+1)
            x = con(x, edge_index, self.alpha, _hidden[0], beta)
            if self.dataset == 'ogbn-arxiv':
                x = self.layers_bn[i](x)
            x = F.relu(x)
            if self.dataset == 'ogbn-arxiv':
                x += _hidden[-1]
            _hidden.append(x)


        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[-1](x)
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

        _hidden = []
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.convs[0](x)
        if self.dataset == 'ogbn-arxiv':
            x = self.bn_input(x)
        x = F.relu(x)
        _hidden.append(x)

        # compute energy in the first layer
        energy = self.Dirichlet_energy(x, laplacian_weight)
        energy_list.append(energy)

        for i, con in enumerate(self.convs[1:-1]):
            x = F.dropout(x, self.dropout, training=self.training)
            beta = math.log(self.lamda / (i + 1) + 1)
            x = con(x, edge_index, self.alpha, _hidden[0], beta)
            if self.dataset == 'ogbn-arxiv':
                x = self.layers_bn[i](x)
            x = F.relu(x)
            if self.dataset == 'ogbn-arxiv':
                x += _hidden[-1]
            _hidden.append(x)

            # compute energy in the first layer
            energy = self.Dirichlet_energy(x, laplacian_weight)
            energy_list.append(energy)

        return energy_list

