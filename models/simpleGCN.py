from torch import nn
# from torch_geometric.nn import SGConv
from models.SGC_layer import SGConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj
import torch

class simpleGCN(nn.Module):
    def __init__(self, args):
        super(simpleGCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.cached = self.transductive = args.transductive

        # bn used in dataset 'obgn-arxiv'
        # lin_first used in Coauthor_physics due to space limit
        self.bn = True if args.type_norm == 'batch' else False
        self.lin_first = True if args.dataset == 'Coauthor_Physics' else False
        self.SGC = SGConv(self.num_feats, self.num_classes, K=self.num_layers,
                          cached=self.cached, bias=False, bn=self.bn, dropout=0.,
                          lin_first=self.lin_first)

    def Dirichlet_energy(self, x, adj):
        # x = F.normalize(x, p=2, dim=1)
        x = torch.matmul(torch.matmul(x.t(), adj), x)
        energy = torch.trace(x)
        return energy.item()

    def forward(self, x, edge_index):
        # implemented based on https://github.com/Tiiiger/SGC/blob/master/citation.py
        x = self.SGC(x, edge_index)
        return x

    def compute_energy(self, x, edge_index, device):

        energy_list = []
        edge_weight = None
        edge_index, edge_weight = gcn_norm(
            edge_index, edge_weight, x.size(0), False, dtype=x.dtype)
        adj_weight = to_dense_adj(edge_index, edge_attr=edge_weight)
        num_nodes = x.size(0)
        adj_weight = torch.squeeze(adj_weight, dim=0)
        laplacian_weight = torch.eye(num_nodes, dtype=torch.float, device=device) - adj_weight

        # compute energy in the first layer
        energy = self.Dirichlet_energy(x, laplacian_weight)
        energy_list.append(energy)

        if self.lin_first:
            x = self.SGC.lin(x)

        for k in range(self.num_layers):
            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.SGC.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

            # compute energy in the middle layer
            energy = self.Dirichlet_energy(x, laplacian_weight)
            energy_list.append(energy)

        return energy_list








