from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch



class APPNP(MessagePassing):
    r"""The approximate personalized propagation of neural predictions layer
    from the `"Predict then Propagate: Graph Neural Networks meet Personalized
    PageRank" <https://arxiv.org/abs/1810.05997>`_ paper

    .. math::
        \mathbf{X}^{(0)} &= \mathbf{X}

        \mathbf{X}^{(k)} &= (1 - \alpha) \mathbf{\hat{D}}^{-1/2}
        \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \mathbf{X}^{(k-1)} + \alpha
        \mathbf{X}^{(0)}

        \mathbf{X}^{\prime} &= \mathbf{X}^{(K)},

    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    The adjacency matrix can include other values than :obj:`1` representing
    edge weights via the optional :obj:`edge_weight` tensor.

    Args:
        K (int): Number of iterations :math:`K`.
        alpha (float): Teleport probability :math:`\alpha`.
        dropout (float, optional): Dropout probability of edges during
            training. (default: :obj:`0`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, args, cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(APPNP, self).__init__(**kwargs)

        self.add_self_loops = add_self_loops
        self.normalize = normalize

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.dim_hidden = args.dim_hidden
        self.adj_dropout = args.adj_dropout
        self.embedding_dropout = args.dropout
        self.beta = args.beta if self.dataset!='ogbn-arxiv' else 0.2
        self.cached = args.transductive
        self.input_trans = torch.nn.Linear(self.num_feats, self.dim_hidden)
        self.output_trans = torch.nn.Linear(self.dim_hidden, self.num_classes)
        self.type_norm = args.type_norm

        if self.type_norm == 'batch':
            self.input_bn = torch.nn.BatchNorm1d(self.dim_hidden)
            self.layers_bn = torch.nn.ModuleList([])
            for _ in range(self.num_layers):
                self.layers_bn.append(torch.nn.BatchNorm1d(self.num_classes))


    def reset_parameters(self):
        self._cached_edge_index = None
        self._cached_adj_t = None


    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        # implemented based on: https://github.com/klicperajo/ppnp/blob/master/ppnp/pytorch/ppnp.py
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # input transformation according to the official implementation
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        x = self.input_trans(x)
        if self.type_norm == 'batch':
            x = self.input_bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.embedding_dropout, training=self.training)
        h = self.output_trans(x)
        x = h

        for k in range(self.num_layers):
            if self.adj_dropout > 0 and self.training:
                if isinstance(edge_index, Tensor):
                    assert edge_weight is not None
                    edge_weight = F.dropout(edge_weight, p=self.adj_dropout, training=self.training)
                else:
                    value = edge_index.storage.value()
                    assert value is not None
                    value = F.dropout(value, p=self.adj_dropout, training=self.training)
                    edge_index = edge_index.set_value(value, layout='coo')

            # propagate_type: (x: Tensor, edge_weight: OptTensor)
            x = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                               size=None)
            if self.type_norm == 'batch':
                x = self.layers_bn[k](x)
            x = x * (1 - self.beta)
            x += self.beta * h

        return x

    def message(self, x_j: Tensor, edge_weight: Tensor) -> Tensor:
        return edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}(K={}, alpha={})'.format(self.__class__.__name__, self.K,
                                           self.alpha)