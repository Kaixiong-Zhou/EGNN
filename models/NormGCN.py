from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from models.common_blocks import act_map, batch_norm
from torch_geometric.nn.inits import glorot, zeros
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import scatter_
import numpy as np
import random

def cluster_index(data, adj, num_cluster=5):

    set_used = set([])
    set_clusters = []
    for _ in range(num_cluster):
        set_clusters.append(set([]))
    len_clusters = [0] * num_cluster
    flag_noNeighbors = [False] * num_cluster
    hops = [0] * num_cluster
    num_hops = 2
    set_allNodes = set(list(range(0, adj.shape[0])))
    num_used= 0.

    train_mask = data.train_mask.data.cpu().numpy()
    data_y = data.y.data.cpu().numpy()
    train_nodes = np.argwhere(train_mask == True).squeeze()
    y_label = data_y[train_mask]

    for i in range(num_cluster):
        index_label = np.argwhere(y_label == i).squeeze()
        nodes_label = train_nodes[index_label]
        adj[:, nodes_label] = 0
        set_clusters[i].update(set(nodes_label))
        set_used.update(set(nodes_label))


    for j in range(num_hops):
        for i in range(num_cluster):
            cluster = list(set_clusters[i])
            neighbors = adj[cluster]
            neighbors = np.argwhere(neighbors > 0)
            neighbors = set(neighbors[:,1])
            neighbors.difference_update(set_used)
            set_used.update(neighbors)
            set_clusters[i].update(neighbors)
            adj[:, list(neighbors)] = 0

    data_x = data.x.data.cpu().numpy()
    set_allNodes.difference_update(set_used)
    centerX_cluster = []
    for i in range(num_cluster):
        x_cluster = np.mean(data_x[list(set_clusters[i])], axis=0)
        centerX_cluster.append(x_cluster)
    centerX_cluster = np.stack(centerX_cluster, axis=0)
    norm_f2 = np.linalg.norm(centerX_cluster, axis=1)
    centerX_cluster = centerX_cluster / np.expand_dims(norm_f2, axis=1)
    centerX_cluster = centerX_cluster.transpose()


    nodes_remain = np.array(list(set_allNodes))
    x_remain = data_x[nodes_remain]
    dis = np.matmul(x_remain, centerX_cluster)
    clusters = np.argmax(dis, axis=1)
    for i in range(num_cluster):
        index_cluster = np.argwhere(clusters == i).squeeze()
        nodes_cluster = nodes_remain[index_cluster]
        set_used.update(set(nodes_cluster))
        set_clusters[i].update(set(nodes_cluster))
        num_used += len(set_clusters[i])
        print(len(set_clusters[i]))

        print(data_y[list(set_clusters[i])])

    print(num_used)

    return set_clusters


class NormGCN(nn.Module):
    def __init__(self, args, num_nodes, adj, data):
        super(NormGCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.batch_normal = args.batch_normal
        self.residual = args.residual
        self.activation = args.activation
        self.dropout = args.dropout
        self.num_neighbors = args.num_neighbors
        self.cached = self.transductive = args.transductive
        self.norm_weight = None
        self.probs_node = None
        self.aggr = 'add'
        self.layers_GCN = nn.ModuleList([])
        self.layers_activation = nn.ModuleList([])
        self.layers_bn = nn.ModuleList([])
        self.layers_residual = nn.ModuleList([])
        # self.num_classes # larger number of clusters would be better in small layer (20/15 in layer 1/2)
        self.num_groups = 20
        self.group_index = []
        # group_nodes = int(num_nodes / self.num_groups)
        # for group in range(self.num_groups):
        #     if group < self.num_groups-1:
        #         self.group_index.append(list(range(group*group_nodes, (group+1)*group_nodes, 1)))
        #     else:
        #         self.group_index.append(list(range(group * group_nodes, num_nodes, 1)))
        group_index = cluster_index(data, adj, self.num_groups)
        num_nodes = adj.shape[0]
        for i in range(self.num_groups):
            score_cluster = torch.zeros((num_nodes,1), dtype=torch.float)
            score_cluster[list(group_index[i])] = 1.
            self.group_index.append(score_cluster)

        if self.num_layers == 1:
            self.layers_GCN.append(GCNConv(self.num_feats, self.num_classes, cached=self.cached, bias=False))
        elif self.num_layers == 2:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))
        else:
            self.layers_GCN.append(GCNConv(self.num_feats, self.dim_hidden, cached=self.cached, bias=False))
            for _ in range(self.num_layers-2):
                self.layers_GCN.append(GCNConv(self.dim_hidden, self.dim_hidden, cached=self.cached, bias=False))
            self.layers_GCN.append(GCNConv(self.dim_hidden, self.num_classes, cached=self.cached, bias=False))


        for i in range(self.num_layers):
            self.layers_activation.append(act_map(self.activation))
            dim_in = self.layers_GCN[i].in_channels
            dim_out = self.layers_GCN[i].out_channels
            if self.batch_normal:
                layers_bn = nn.ModuleList([])
                for group in range(self.num_groups):
                    layers_bn.append(batch_norm(dim_out))
                self.layers_bn.append(layers_bn)

            if self.residual:
                self.layers_residual.append(nn.Linear(dim_in, dim_out, bias=False))

        self.final_bn = nn.ModuleList([])
        for group in range(self.num_groups):
            self.final_bn.append(batch_norm(self.num_classes))
        self.final_cluster = torch.nn.Parameter(torch.Tensor(self.num_classes, self.num_groups))
        glorot(self.final_cluster)

    def forward(self, x, edge_index):

        layers = [x]
        for i in range(self.num_layers):
            if i == 0 or i == self.num_layers-1:
                x = F.dropout(x, p=self.dropout, training=self.training)
            x_conv = self.layers_GCN[i](x, edge_index)
            if self.batch_normal:
                # x_mean = x_conv.mean(dim=0)
                # x_conv = x_conv - x_mean
                # rownorm_mean = (1e-6 + x_conv.pow(2).sum(dim=1).mean()).sqrt()
                # x_conv =  x_conv / rownorm_mean

                running_mean = torch.stack([self.layers_bn[i][group].bn.running_mean
                                            for group in range(self.num_groups)], dim=0).t()
                score_cluster = F.softmax(torch.mm(x_conv, running_mean), dim=1)
                x_temp = []
                for group in range(self.num_groups):
                    tmp = score_cluster[:, group].unsqueeze(dim=1) * x_conv
                    # tmp = self.group_index[group] * x_conv
                    tmp = self.layers_bn[i][group](tmp)
                    x_temp.append(tmp)
                x_conv = torch.sum(torch.stack(x_temp, dim=2), dim=2)


            # normalization in final layer
            if i == self.num_layers-1:
                running_mean = torch.stack([self.final_bn[group].bn.running_mean
                                            for group in range(self.num_groups)], dim=0).t()
                score_cluster = F.softmax(torch.mm(x_conv, running_mean), dim=1)
                # score_cluster = F.softmax(torch.mm(x_conv, self.final_cluster), dim=1)
                x_temp = []
                for group in range(self.num_groups):
                    tmp = score_cluster[:, group].unsqueeze(dim=1) * x_conv
                    # tmp = self.group_index[group] * x_conv
                    tmp = self.final_bn[group](tmp)
                    x_temp.append(tmp)
                x_temp = torch.sum(torch.stack(x_temp, dim=2), dim=2)
                x_conv = x_conv + x_temp * 0.005

            x_conv = self.layers_activation[i](x_conv)
            if self.residual:
                x = x_conv + self.layers_residual[i](x)
            else:
                x = x_conv
            layers.append(x)

        return x, layers

    def extract_neighbor(self, x, edge_index, norm, label):
        size_neighbor = edge_index.size(1) - x.size(0)
        edge_index_neighbor = edge_index[:, :size_neighbor]
        norm_neighbor = norm.view(-1, 1)[:size_neighbor, :]
        # print(norm)
        x_j = x.index_select(0, edge_index_neighbor[0])

        # all neighbors
        x_j_all = scatter_(self.aggr, norm_neighbor*x_j, edge_index_neighbor[1], 0, x.size(0))
        # neighbors of the same or different label
        label_source = label.index_select(0, edge_index_neighbor[0])
        label_target = label.index_select(0, edge_index_neighbor[1])
        norm_pos = torch.zeros((norm_neighbor.size(0),1), dtype=norm.dtype,
                                 device=norm.device)
        norm_neg = torch.zeros((norm_neighbor.size(0), 1), dtype=norm.dtype,
                               device=norm.device)
        norm_pos[label_source==label_target] = norm_neighbor[label_source==label_target]
        norm_neg[label_source!=label_target] = norm_neighbor[label_source!=label_target]

        x_j_pos = scatter_(self.aggr, norm_pos*x_j, edge_index_neighbor[1], 0, x.size(0))
        x_j_neg = scatter_(self.aggr, norm_neg*x_j, edge_index_neighbor[1], 0, x.size(0))
        # print(norm_pos, norm_neg, torch.sum(norm), torch.sum(norm_pos), torch.sum(norm_neg))
        return x_j_all, x_j_pos, x_j_neg

    def norm(self, x, edge_index):
        edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype,
                                 device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def get_probs_node(self, x, edge_index):
        if self.probs_node is not None:
            return self.probs_node
        else:
            edge_weight = torch.ones((edge_index.size(1), 1), dtype=x.dtype,
                                     device=edge_index.device)
            row, col = edge_index
            deg = scatter_add(edge_weight, row, dim=0, dim_size=x.size(0))
            probs_node = deg / edge_index.shape[1]
            self.probs_node = probs_node
            return probs_node


    def extract_actitity(self, x, edge_index, label):
        layers_self = []
        layers_neighbor = []
        layers_neighbor_pos = []
        layers_neighbor_neg = []
        if self.norm_weight is None:
            self.norm_weight = self.norm(x, edge_index)
        norm = self.norm_weight

        _, layers =self.forward(x, edge_index)
        for i in range(self.num_layers+1):
            x_j_all, x_j_pos, x_j_neg = self.extract_neighbor(layers[i], edge_index, norm, label)
            layers_self.append(layers[i])
            layers_neighbor.append(x_j_all)
            layers_neighbor_pos.append(x_j_pos)
            layers_neighbor_neg.append(x_j_neg)

        return layers_self, layers_neighbor, layers_neighbor_pos, layers_neighbor_neg





