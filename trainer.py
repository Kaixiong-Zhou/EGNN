import torch
import os
from models.simpleGCN import simpleGCN
from models.GAT import GAT
from models.EGNN import EGNN
from models.GCNII import GCNII
from models.GCN import GCN
from models.JKNet import JKNetMaxpool
from models.EdgeDrop import EdgeDrop
from models.APPNP import APPNP



from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import PPI
from torch_geometric.datasets import Coauthor, Amazon
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops, dense_to_sparse, to_undirected
import numpy as np
# from torch_geometric.utils import scatter_, to_dense_adj, contains_isolated_nodes
# from torch_sparse import spspmm, coalesce, to_scipy, from_scipy
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# import scipy.sparse
# import matplotlib.gridspec as gridspec
# from options.base_options import reset_weight
from sklearn import metrics



def load_ogbn(dataset='ogbn-arxiv'):
    dataset = PygNodePropPredDataset(name=dataset)
    split_idx = dataset.get_idx_split()
    data = dataset[0]
    data.edge_index = to_undirected(data.edge_index, data.num_nodes)
    return data, split_idx

def load_data(dataset="Cora"):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', dataset)
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        data = Planetoid(path, dataset, split='public', transform=T.NormalizeFeatures())[0]
        # data = Planetoid(path, dataset, transform=T.NormalizeFeatures())[0]

        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index
        return data

    elif dataset in ['CoauthorCS', 'Coauthor_Physics', 'Amazon_Computers']:
        if dataset in ['CoauthorCS', 'Coauthor_Physics']:
            file_name = 'CS' if dataset=='CoauthorCS' else "Physics"
            data = Coauthor(path, file_name, T.NormalizeFeatures())[0]
        else:
            file_name = 'Computers' if dataset == 'Amazon_Computers' else "Photo"
            data = Amazon(path, file_name, T.NormalizeFeatures())[0]

        num_nodes = data.x.size(0)
        edge_index, _ = remove_self_loops(data.edge_index)
        edge_index = add_self_loops(edge_index, num_nodes=num_nodes)
        if isinstance(edge_index, tuple):
            data.edge_index = edge_index[0]
        else:
            data.edge_index = edge_index

        # # devide training validation and testing set
        # train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        # val_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        # test_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        # train_num = 20
        # val_num = 30
        # if dataset in ['CoauthorCS', 'Coauthor_Physics']:
        #     num_classes = 15 if dataset=='CoauthorCS' else 5
        # else:
        #     num_classes = 10 if dataset == 'Amazon_Computers' else 8
        # for i in range(num_classes): # number of labels
        #     index = (data.y == i).nonzero()[:,0]
        #     perm = torch.randperm(index.size(0))
        #     # print(index[perm[:train_num]])
        #     # print(perm[train_num:(train_num+val_num)])
        #     # print(index[perm[(train_num+val_num):]])
        #     train_mask[index[perm[:train_num]]] = 1
        #     val_mask[index[perm[train_num:(train_num+val_num)]]] = 1
        #     test_mask[index[perm[(train_num+val_num):]]] = 1
        # data.train_mask = train_mask
        # data.val_mask = val_mask
        # data.test_mask = test_mask
        return data
    else:
        raise Exception(f'the dataset of {dataset} has not been implemented')

def load_ppi_data():
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'PPI')
    train_dataset = PPI(path, split='train', transform=T.NormalizeFeatures())
    val_dataset = PPI(path, split='val', transform=T.NormalizeFeatures())
    test_dataset = PPI(path, split='test', transform=T.NormalizeFeatures())
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    return [train_loader, val_loader, test_loader]


def evaluate(output, labels, mask):
    _, indices = torch.max(output, dim=1)
    correct = torch.sum(indices[mask] == labels[mask])
    return correct.item() * 1.0 / mask.sum().item()

class trainer(object):
    def __init__(self, args, train_mask=None, val_mask=None, test_mask=None):
        self.args = args
        self.dataset = args.dataset
        self.device = torch.device(f'cuda:{args.cuda_num}' if args.cuda else 'cpu')
        if self.dataset in ["Cora", "Citeseer", "Pubmed", 'CoauthorCS', 'Coauthor_Physics', 'Amazon_Computers']:
            self.data = load_data(self.dataset)
            if self.dataset in ['CoauthorCS', 'Coauthor_Physics', 'Amazon_Computers']:
                self.data.train_mask = train_mask
                self.data.val_mask = val_mask
                self.data.test_mask = test_mask
            self.loss_fn = torch.nn.functional.nll_loss
            self.data.to(self.device)
        elif self.dataset in ['PPI']:
            self.data = load_ppi_data()
            self.loss_fn = torch.nn.BCEWithLogitsLoss()
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
        elif self.type_model == 'simpleGCN':
            self.model = simpleGCN(args)
        elif self.type_model in ['GCN', 'pairnorm']:
            if args.type_model == 'pairnorm':
                args.type_norm = 'pair'
            self.model = GCN(args)
        elif self.type_model == 'JKNet':
            self.model = JKNetMaxpool(args)
        elif self.type_model == 'EdgeDrop':
            self.model = EdgeDrop(args)
        elif self.type_model == 'GAT':
            self.model = GAT(args)
        elif self.type_model == 'APPNP':
            self.model = APPNP(args)

        self.model.to(self.device)

        self.weight_decay1 = args.weight_decay1 # 0.
        # self.weight_decay1 = args.weight_decay2 # for testing Glorot

        self.weight_decay2 = args.weight_decay2 # 5e-4
        self.weight_decay3 = args.weight_decay3 # 1e-4/32,64 1e-1/16 in iso-residual-GCN
        self.loss_weight = args.loss_weight
        self.c_max = args.c_max

        if self.type_model == 'EGNN':
            self.optimizer = torch.optim.Adam([dict(params=self.model.reg_params, weight_decay=self.weight_decay1),
            dict(params=self.model.non_reg_params, weight_decay=self.weight_decay2),
            dict(params=self.model.srelu_params, weight_decay=self.weight_decay3)], lr=args.lr)
        elif self.type_model == 'GCNII':
            self.optimizer = torch.optim.Adam([
                dict(params=self.model.reg_params, weight_decay=args.wd1),
                dict(params=self.model.non_reg_params, weight_decay=args.wd2)
            ], lr=args.lr)

        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=args.lr, weight_decay=self.weight_decay2)


    def train_and_test(self, compute_energy=False):
        best_val_acc = 0.
        best_train_loss = 100
        best_test_acc = 0.
        best_train_acc = 0.
        best_val_loss = 100.
        patience = self.args.patience
        bad_counter = 0.
        val_loss_history = []

        for epoch in range(self.epochs):
            loss_train, acc_train, acc_valid, acc_test, loss_val = self.train_net()
            val_loss_history.append(loss_val)


            if self.dataset != 'ogbn-arxiv':
                if loss_val < best_val_loss:
                    best_val_loss = loss_val
                    best_test_acc = acc_test
                    best_val_acc = acc_valid
                    best_train_loss = loss_train
                    best_train_acc = acc_train
                    bad_counter = 0
                    if compute_energy:
                        self.model.cpu()
                        self.save_model()
                        self.model.to(self.device)
                else:
                    bad_counter += 1
            else:
                if acc_valid > best_val_acc:
                    best_val_loss = loss_val
                    best_test_acc = acc_test
                    best_val_acc = acc_valid
                    best_train_loss = loss_train
                    best_train_acc = acc_train
                    bad_counter = 0
                    if compute_energy:
                        self.model.cpu()
                        self.save_model()
                        self.model.to(self.device)
                else:
                    bad_counter += 1

            if epoch % 20 == 0:
                log = 'Epoch: {:03d}, Train loss: {:.4f}, Val loss: {:.4f}, Test acc: {:.4f}'
                print(log.format(epoch, loss_train, loss_val, acc_test))
            if bad_counter == patience:
                break

        print('val_acc: {:.4f}, test_acc:{:.4f}'.format(best_val_acc, best_test_acc))

        # compute energy. now only for EGNN
        if compute_energy:
            state_dict = self.load_model()
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            with torch.no_grad():
                energy_list = self.model.compute_energy(self.data.x, self.data.edge_index,
                                                        self.device)

        if compute_energy:
            print('energy list: ', energy_list)
            return energy_list
        else:
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
        loss = 0.
        if self.dataset in ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS',
                            'Coauthor_Physics', 'Amazon_Computers']:
            raw_logits = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(raw_logits[self.data.train_mask], 1)
            loss = self.loss_fn(logits, self.data.y[self.data.train_mask])
        elif self.dataset in ['PPI']:
            for data in self.data[0]:
                num_nodes = data.x.size(0)
                # edge_index, _ = remove_self_loops(data.edge_index)
                data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)
                if isinstance(data.edge_index, tuple):
                    data.edge_index = data.edge_index[0]
                logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                loss += self.loss_fn(logits, data.y.to(self.device))
        elif self.dataset == 'ogbn-arxiv':
            pred = self.model(self.data.x, self.data.edge_index)
            pred = F.log_softmax(pred[self.train_idx], 1)
            loss = self.loss_fn(pred, self.data.y.squeeze(1)[self.train_idx])
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


            loss = loss + self.loss_weight * loss_orthogonal
            # iso-residual-GCN: 5e-1/16, 5./32, 10./64
            # 2e-1 in iso-att-GCN and 5e-1 in iso-GCN, iso-residual-GCN

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run_testSet(self):
        self.model.eval()
        # torch.cuda.empty_cache()
        if self.dataset in ['Cora', 'Citeseer', 'Pubmed', 'CoauthorCS',
                            'Coauthor_Physics', 'Amazon_Computers']:
            with torch.no_grad():
                logits = self.model(self.data.x, self.data.edge_index)
            logits = F.log_softmax(logits, 1)
            acc_train = evaluate(logits, self.data.y, self.data.train_mask)
            acc_valid = evaluate(logits, self.data.y, self.data.val_mask)
            acc_test = evaluate(logits, self.data.y, self.data.test_mask)
            val_loss = self.loss_fn(logits[self.data.val_mask], self.data.y[self.data.val_mask])
            return acc_train, acc_valid, acc_test, val_loss
        elif self.dataset in ['PPI']:
            accs = [0., 0., 0.]
            for i in range(1, 3):
                total_micro_f1 = 0.
                for data in self.data[i]:
                    num_nodes = data.x.size(0)
                    # edge_index, _ = remove_self_loops(data.edge_index)
                    data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)
                    if isinstance(data.edge_index, tuple):
                        data.edge_index = data.edge_index[0]
                    with torch.no_grad():
                        logits = self.model(data.x.to(self.device), data.edge_index.to(self.device))
                    pred = (logits > 0).float().cpu()
                    micro_f1 = metrics.f1_score(data.y, pred, average='micro')
                    total_micro_f1 += micro_f1
                total_micro_f1 /= len(self.data[i].dataset)
                accs[i] = total_micro_f1
            return accs[0], accs[1], accs[2], 0.
        elif self.dataset == 'ogbn-arxiv':
            out = self.model(self.data.x, self.data.edge_index)
            out = F.log_softmax(out, 1)
            y_pred = out.argmax(dim=-1, keepdim=True)

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

            return train_acc, valid_acc, test_acc, 0.

        else:
            raise Exception(f'the dataset of {self.dataset} has not been implemented')




    def filename(self, filetype='params'):
        filedir = f'./{filetype}/{self.dataset}'
        if not os.path.exists(filedir):
            os.makedirs(filedir)

        c_min = int(self.args.c_min * 100)
        c_max = int(self.args.c_max * 100)
        loss_weight = int(self.args.loss_weight)
        bias_SReLU = int(self.args.bias_SReLU)
        filename = self.args.type_model + f'_layer{self.num_layers}_{c_min}_{c_max}_{loss_weight}_{bias_SReLU}.pth.tar'
        filename = os.path.join(filedir, filename)

        return filename


    def load_model(self):
        filename = self.filename(filetype='params')
        if os.path.exists(filename):
            print('load model: ', self.args.type_model, filename)
            return torch.load(filename)
        else:
            return None

    def save_model(self):
        filename = self.filename(filetype='params')
        state = self.model.state_dict()
        torch.save(state, filename)
        # print('save model to', filename)

