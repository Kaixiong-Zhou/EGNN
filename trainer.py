import torch
import os
from models.simpleGCN import simpleGCN
from models.EGNN import EGNN
from models.GCNII import GCNII
from models.GCN import GCN
from models.JKNet import JKNetMaxpool
from models.EdgeDrop import EdgeDrop
from models.APPNP import APPNP

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import Coauthor, Amazon
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, dense_to_sparse, to_undirected
import numpy as np
from sklearn import metrics


def load_ogbn(dataset='ogbn-arxiv'):
    dataset = PygNodePropPredDataset(name=dataset)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    return data, split_idx

def load_data(dataset="Cora"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["Cora", "Pubmed"]:
        data = Planetoid(path, dataset, split='public', transform=T.NormalizeFeatures())[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index
        return data

    elif dataset == 'Coauthor_Physics':
        data = Coauthor(path, "Physics", T.NormalizeFeatures())[0]
        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index
        return data
    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')


def evaluate(y_pred, labels, mask):
    correct = torch.sum(y_pred[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()


class trainer(object):
    def __init__(self, args, train_mask=None, val_mask=None, test_mask=None):
        self.args = args
        self.dataset = args.dataset
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        if self.dataset in ["Cora", "Pubmed", 'Coauthor_Physics']:
            self.data = load_data(self.dataset)
            if self.dataset == 'Coauthor_Physics':
                self.data.train_mask = train_mask
                self.data.val_mask = val_mask
                self.data.test_mask = test_mask
            self.loss_fn = torch.nn.functional.nll_loss
            self.data.to(self.device)
        elif self.dataset == 'ogbn-arxiv':
            self.data, self.split_idx = load_ogbn(self.dataset)
            self.data.to(self.device)
            self.train_idx = self.split_idx['train'].to(self.device)
            self.evaluator = Evaluator(name='ogbn-arxiv')
            self.loss_fn = torch.nn.functional.nll_loss
        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')

        self.type_model = args.type_model
        self.epochs = args.epochs
        self.num_layers = args.num_layers
        self.dim_hidden = args.dim_hidden
        if self.type_model == 'EGNN':
            self.model = EGNN(args)
        elif self.type_model == 'GCNII':
            self.model = GCNII(args)
        elif self.type_model == 'SGC':
            self.model = simpleGCN(args)
        elif self.type_model in ['GCN', 'pairnorm']:
            if args.type_model == 'pairnorm':
                args.type_norm = 'pair'
            self.model = GCN(args)
        elif self.type_model == 'JKNet':
            self.model = JKNetMaxpool(args)
        elif self.type_model == 'EdgeDrop':
            self.model = EdgeDrop(args)
        elif self.type_model == 'APPNP':
            self.model = APPNP(args)
        else:
            raise Exception(f'the model of {self.type_model} has not been implemented')

        self.model.to(self.device)
        self.loss_weight = args.loss_weight
        self.c_max = args.c_max

        if self.type_model == 'EGNN':
            self.optimizer = torch.optim.Adam([dict(params=self.model.reg_params, weight_decay=0.),
            dict(params=self.model.non_reg_params, weight_decay=args.weight_decay),
            dict(params=self.model.srelu_params, weight_decay=args.weight_decay_shift)], lr=args.lr)
        elif self.type_model == 'GCNII':
            self.optimizer = torch.optim.Adam([
                dict(params=self.model.reg_params, weight_decay=args.GCNII_wd1),
                dict(params=self.model.non_reg_params, weight_decay=args.GCNII_wd2)
            ], lr=args.lr)

        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=args.lr, weight_decay=args.weight_decay)


    def train_and_test(self):
        best_val_acc = 0.
        best_test_acc = 0.
        best_val_loss = 100.
        patience = self.args.patience
        bad_counter = 0.

        for epoch in range(self.epochs):
            loss_train, acc_train, acc_valid, acc_test, loss_val = self.train_net()

            if self.dataset != 'ogbn-arxiv':
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_test_acc = acc_test
                    best_val_acc = acc_valid
                    bad_counter = 0
                else:
                    bad_counter += 1
            else:
                if acc_valid > best_val_acc:
                    best_val_loss = loss_val
                    best_test_acc = acc_test
                    best_val_acc = acc_valid
                    bad_counter = 0
                else:
                    bad_counter += 1

            if epoch % 20 == 0:
                log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:.4f}'
                print(log.format(epoch, loss_train, acc_valid, acc_test))
            if bad_counter == patience:
                break

        print('val_acc: {:.4f}, test_acc:{:.4f}'.format(best_val_acc, best_test_acc))
        return best_test_acc

    def train_net(self):
        try:
            loss_train = self.run_trainSet()
            acc_train, acc_valid, acc_test, loss_val = self.run_testSet()
            return loss_train, acc_train, acc_valid, acc_test, loss_val
        except RuntimeError as e:
            if "cuda" in str(e) or "CUDA" in str(e):
                print(e)
            else:
                raise e


    def run_trainSet(self):
        self.model.train()
        out = self.model(self.data.x, self.data.edge_index)

        if self.dataset in ['Cora', 'Pubmed', 'Coauthor_Physics']:
            logits = F.log_softmax(out[self.data.train_mask], 1)
            loss = self.loss_fn(logits, self.data.y[self.data.train_mask])
        elif self.dataset == 'ogbn-arxiv':
            logits = F.log_softmax(out[self.train_idx], 1)
            loss = self.loss_fn(logits, self.data.y.squeeze(1)[self.train_idx])
        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')


        if self.type_model == 'EGNN':
            weight_standard = torch.eye(self.dim_hidden, device=self.device)
            weight_first_layer = torch.eye(self.dim_hidden, device=self.device) * \
                                 np.sqrt(self.c_max)
            loss_orthogonal = 0.
            loss_orthogonal += torch.norm(self.model.layers_GCN[0].weight - weight_first_layer)
            for i in range(1, self.model.num_layers):
                loss_orthogonal += torch.norm(self.model.layers_GCN[i].weight - weight_standard)
            loss += self.loss_weight * loss_orthogonal

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    @torch.no_grad()
    def run_testSet(self):
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        logits = F.log_softmax(out, 1)
        y_pred = out.argmax(dim=-1, keepdim=True)
        if self.dataset in ['Cora',  'Pubmed', 'Coauthor_Physics']:
            y_pred = y_pred.squeeze()
            acc_train = evaluate(y_pred, self.data.y, self.data.train_mask)
            acc_valid = evaluate(y_pred, self.data.y, self.data.val_mask)
            acc_test = evaluate(y_pred, self.data.y, self.data.test_mask)
            val_loss = self.loss_fn(logits[self.data.val_mask], self.data.y[self.data.val_mask])
            return acc_train, acc_valid, acc_test, val_loss

        elif self.dataset == 'ogbn-arxiv':
            train_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['train']],
                'y_pred': y_pred[self.split_idx['train']],
            })['acc']
            valid_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['valid']],
                'y_pred': y_pred[self.split_idx['valid']],
            })['acc']
            test_acc = self.evaluator.eval({
                'y_true': self.data.y[self.split_idx['test']],
                'y_pred': y_pred[self.split_idx['test']],
            })['acc']

            return train_acc, valid_acc, test_acc, None

        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')


