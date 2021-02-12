import torch
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops # add_remaining_self_loops
import math

from torch_geometric.nn.inits import glorot, zeros
import math
def othogonal(tensor):
    if tensor is not None:
        tensor.data = torch.eye(tensor.size(-1))


class GCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classification with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, c_min=0.5, c_max=1.,
                 improved=False, cached=False, bias=True,  **kwargs):
        super(GCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.residual_weight = float(c_min / c_max)

        if in_channels != out_channels:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            # self.weight = Parameter(torch.zeros((in_channels, out_channels)))
            # self.weight.data.fill_diagonal_(math.sqrt(c_max)) # 0.92
            self.weight = Parameter(torch.eye(in_channels) * math.sqrt(c_max))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        # self.diagonal_constant = torch.eye(in_channels, device=torch.device(f'cuda:4')) * math.sqrt(c_max)

        self.reset_parameters()

    def reset_parameters(self):
        if self.in_channels != self.out_channels:
            glorot(self.weight)
            # self.weight.data += torch.eye(self.in_channels, self.out_channels)
        zeros(self.bias)
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        # fill_value = 1 if not improved else 2
        # edge_index, edge_weight = add_remaining_self_loops(
        #     edge_index, edge_weight, fill_value, num_nodes)
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype,
                                 device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, x_0 = None, alpha=0., beta=0.,
                residual_weight=0., conv_mode='GCN', edge_weight=None):
        """"""
        x_input = x
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}. Please '
                    'disable the caching behavior of this layer by removing '
                    'the `cached=True` argument in its constructor.'.format(
                        self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight,
                                         self.improved, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        x = self.propagate(edge_index, x=x, norm=norm)

        if conv_mode == 'iso-GCNII':
            x = (1-residual_weight-alpha) * x + residual_weight * x_input \
                + alpha * x_0
            x = (1 - beta) * x + beta * torch.matmul(x, self.weight)
        elif conv_mode in ['iso-GCN', 'GCN', 'iso-att-GCN']:
            x = torch.matmul(x, self.weight)
            # x = (1 - beta) * torch.matmul(x, self.diagonal_constant) + beta * torch.matmul(x, self.weight)
        elif conv_mode == 'iso-residual-GCN':
            x = (1-residual_weight-alpha) * x + residual_weight * x_input \
                + alpha * x_0
            x = torch.matmul(x, self.weight)
        else:
            raise Exception(f'convlution mode of {conv_mode} has not been implemented')
        return x


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
