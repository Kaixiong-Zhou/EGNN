import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class JKNetMaxpool(torch.nn.Module):
    """An implementation of Jumping Knowledge Network (arxiv 1806.03536) which
    combine layers with Maxpool.
    Args:
        in_features (int): Size of each input node.
        out_features (int): Size of each output node.
        n_layers (int): Number of the convolution layers.
        n_units (int): Size of the mtorchiddle layers.
        aggregation (str): 'sum', 'mean' or 'max'.
                           Specify the way to aggregate the neighbourhoods.
    """
    def __init__(self, args):
        super(JKNetMaxpool, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.dropout = args.dropout
        self.cached = args.transductive
        self.layers_GCN = torch.nn.ModuleList([])
        self.layers_bn = torch.nn.ModuleList([])
        self.type_norm = args.type_norm

        self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached))
        if self.type_norm == 'batch':
            self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))


        for i in range(1, self.num_layers):
            self.layers_GCN.append(
                GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached))

            if self.type_norm == 'batch':
                self.layers_bn.append(torch.nn.BatchNorm1d(self.dim_hidden))

        self.last_linear = torch.nn.Linear(self.dim_hidden, self.num_classes)

    def forward(self, x, edge_index):
        # implemented based on: https://github.com/ShinKyuY/Representation_Learning_on_Graphs_with_Jumping_Knowledge_Networks/blob/master/JK-Net.ipynb

        layer_outputs = []
        for i in range(self.num_layers):
            x = self.layers_GCN[i](x, edge_index)
            if self.type_norm == 'batch':
                x = self.layers_bn[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            layer_outputs.append(x)

        h = torch.stack(layer_outputs, dim=0)
        h = torch.max(h, dim=0)[0]
        x = self.last_linear(h)
        return x

