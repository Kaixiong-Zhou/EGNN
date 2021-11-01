from torch import nn
from models.SGC_layer import SGConv
import torch

class simpleGCN(nn.Module):
    def __init__(self, args):
        super(simpleGCN, self).__init__()
        self.dataset = args.dataset
        self.num_layers = args.num_layers
        self.num_feats = args.num_feats
        self.num_classes = args.num_classes
        self.cached = args.transductive

        # batch normalization used in dataset 'obgn-arxiv'
        # lin_first used in Coauthor_physics due to space limit
        self.bn = True if args.type_norm == 'batch' else False
        self.lin_first = True if args.dataset == 'Coauthor_Physics' else False
        self.SGC = SGConv(self.num_feats, self.num_classes, K=self.num_layers,
                          cached=self.cached, bias=False, bn=self.bn, dropout=0.,
                          lin_first=self.lin_first)

    def forward(self, x, edge_index):
        # implemented based on https://github.com/Tiiiger/SGC/blob/master/citation.py
        x = self.SGC(x, edge_index)
        return x






