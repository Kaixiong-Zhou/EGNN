from torch import nn
import torch
import torch.nn.functional as F
from models.EGNN_layer import EGNNConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import remove_self_loops
from torch_scatter import scatter_add

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
        self.dropout = args.dropout
        self.cached = self.transductive = args.transductive
        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.c_min = args.c_min
        self.c_max = args.c_max # 1. for iso-GCNII, iso-att-GCN; 1.2 for iso-GCN, 1.05 16/1. 32 64 in iso-residual-GCN
        self.bias_SReLU = args.bias_SReLU
        self.output_dropout = args.output_dropout
        self.alpha = args.alpha

        self.reg_params = []
        for i in range(self.num_layers):
            c_min = self.c_min
            c_max = self.c_max if i==0 else 1.
            self.layers_GCN.append(EGNNConv(self.dim_hidden, self.dim_hidden, c_min=c_min,
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
        if self.dataset in ['ogbn-arxiv']:
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.input_layer2(x)
        else:
            x = F.relu(x)

        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.alpha
            x = self.layers_GCN[i](x, edge_index, original_x,
                                   alpha=self.alpha, residual_weight=residual_weight)

            if self.dataset == 'ogbn-arxiv':
                x = self.layers_bn[i](x)

            x = self.layers_activation[i](x)  # used in dropout, 85.0
            # x = F.relu(x)

        x = F.dropout(x, p=self.output_dropout, training=self.training) # 0.2
        x = self.output_layer(x)

        return x


    def Dirichlet_energy(self, x, adj):
        # x = F.normalize(x, p=2, dim=1)
        x = torch.matmul(torch.matmul(x.t().cpu(), adj), x.cpu())
        energy = torch.trace(x)

        return energy.item()

    def Dirichlet_energy_large_graph(self, x, row, col, norm_row, norm_col):
        x_cpu = x.cpu()
        batch_size = 10000
        num_edge = row.size(0)
        num_batch = int(num_edge / batch_size) + 1

        energy = 0.
        for i in range(num_batch-1):

            index = slice(i*batch_size, (i+1)*batch_size)
            x_norm = torch.norm(x_cpu[row[index]]*norm_row[index] - x_cpu[col[index]]*norm_col[index],
                                p='fro', dim=1)
            energy += torch.sum(x_norm).item()

        x_tmp = x_cpu[row[(num_batch-1)*batch_size:]] * norm_row[(num_batch-1)*batch_size:] \
                - x_cpu[col[(num_batch-1)*batch_size:]] * norm_col[(num_batch-1)*batch_size:]

        x_norm = torch.norm(x_tmp, p='fro', dim=1)
        energy += torch.sum(x_norm).item()

        return energy / 2.


    def compute_energy(self, x, edge_index, device):

        energy_list = []
        # edge_weight = None
        # edge_index_cached, edge_weight = gcn_norm(
        #     edge_index, edge_weight, x.size(0), False, dtype=x.dtype)
        # adj_weight = to_dense_adj(edge_index_cached, edge_attr=edge_weight)
        # num_nodes = x.size(0)
        # adj_weight = torch.squeeze(adj_weight, dim=0)
        # laplacian_weight = torch.eye(num_nodes, dtype=torch.float, device=device) - adj_weight

        ## for computing energy of large graph
        num_nodes = x.size(0)
        edge_index_energy, _ = remove_self_loops(edge_index.cpu())
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        edge_weight = torch.ones((edge_index_energy.size(1),), dtype=torch.float,
                                 device=edge_index_energy.device)

        row, col = edge_index_energy[0], edge_index_energy[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg = (1.+deg).pow_(-0.5)
        deg.masked_fill_(deg == float('inf'), 0.)
        norm_row = deg[row].unsqueeze(1)
        norm_col = deg[col].unsqueeze(1)


        # compute energy in the first layer
        # energy = self.Dirichlet_energy(x, laplacian_weight)
        energy = self.Dirichlet_energy_large_graph(x, row, col, norm_row, norm_col)
        energy_list.append(energy)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.input_layer(x)
        if self.dataset in ['ogbn-arxiv']:
            x = self.bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.input_layer2(x)
        else:
            x = F.relu(x)

        # compute energy in the first layer
        # energy = self.Dirichlet_energy(x, laplacian_weight)
        energy = self.Dirichlet_energy_large_graph(x, row, col, norm_row, norm_col)
        energy_list.append(energy)

        original_x = x
        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            residual_weight = self.c_min - self.alpha
            x = self.layers_GCN[i](x, edge_index, original_x,
                                   alpha=self.alpha, residual_weight=residual_weight)

            if self.dataset == 'ogbn-arxiv':
                x = self.layers_bn[i](x)
            x = self.layers_activation[i](x)  # used in dropout, 85.0

            #  compute energy in the middle layer
            # energy = self.Dirichlet_energy(x, laplacian_weight)
            energy = self.Dirichlet_energy_large_graph(x, row, col, norm_row, norm_col)
            energy_list.append(energy)

        return energy_list






