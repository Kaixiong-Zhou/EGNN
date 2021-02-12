import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()

        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.dropout = args.dropout
        self.cached = self.transductive = args.transductive
        self.layers_GCN = torch.nn.ModuleList([])
        self.layers_bn = torch.nn.ModuleList([])
        self.type_norm = args.type_norm

        # space limit
        if self.dataset == 'obgn-arxiv':
            self.dim_hidden = 1

        self.layers_GCN.append(GATConv(self.num_feats, self.dim_hidden,
                                       bias=False, concat=False))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        for _ in range(self.num_layers - 2):
            self.layers_GCN.append(
                GATConv(self.dim_hidden, self.dim_hidden, bias=False, concat=False))
            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        self.layers_GCN.append(GATConv(self.dim_hidden, self.num_classes, bias=False, concat=False))

    def forward(self, x, edge_index):

        for i in range(self.num_layers - 1):
            x = self.layers_GCN[i](x, edge_index)
            if self.type_norm == 'batch':
                x = self.layers_bn[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.layers_GCN[-1](x, edge_index)
        return x