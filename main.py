from options.base_options import BaseOptions, reset_weight
from trainer import trainer
import torch
import os
import numpy as np
import random
import gc
from torch_geometric.datasets import Coauthor, Amazon
import torch_geometric.transforms as T



def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_num)
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

seeds = [123] #  + [100, 123, 111]


def main(args):
    train_mask = None
    val_mask = None
    test_mask = None

    if args.dataset in ['CoauthorCS', 'Coauthor_Physics', 'Amazon_Computers']:
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.dataset)
        if args.dataset in ['CoauthorCS', 'Coauthor_Physics']:
            file_name = 'CS' if args.dataset=='CoauthorCS' else "Physics"
            data = Coauthor('../data', file_name, T.NormalizeFeatures())[0]
        else:
            file_name = 'Computers' if args.dataset == 'Amazon_Computers' else "Photo"
            data = Amazon(path, file_name, T.NormalizeFeatures())[0]

        num_nodes = data.x.size(0)
        train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        val_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        test_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        train_num = 20
        val_num = 30
        for i in range(args.num_classes):  # number of labels
            index = (data.y == i).nonzero()[:, 0]
            perm = torch.randperm(index.size(0))
            train_mask[index[perm[:train_num]]] = 1
            val_mask[index[perm[train_num:(train_num + val_num)]]] = 1
            test_mask[index[perm[(train_num + val_num):]]] = 1
        del data



    if args.type_model == 'EGNN':
        args = reset_weight(args)
    list_acc = []

    for _ in range(10): # 10
        # args.random_seed = seed
        # set_seed(args)
        torch.cuda.empty_cache()
        trnr = trainer(args, train_mask, val_mask, test_mask)
        best_test_acc = trnr.train_and_test(args.compute_energy)
        list_acc.append(best_test_acc)

        del trnr
        torch.cuda.empty_cache()
        gc.collect()

    if args.compute_energy:
        energy_list = np.stack(list_acc, axis=0)
        print('avg energy_list ', np.mean(energy_list, axis=0))
    else:
        print(list_acc)
        print('avg test acc: ', np.mean(list_acc))

    ## record training data
    filedir = f'./logs/{args.dataset}'
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    c_min = int(args.c_min * 100)
    c_max = int(args.c_max * 100)
    loss_weight = int(args.loss_weight*1e6)
    bias_SReLU = int(args.bias_SReLU)

    if args.compute_energy:
        filename = args.type_model + f'energy_layer{args.num_layers}_{c_min}_{c_max}_{loss_weight}_{bias_SReLU}.npy'
        filename = os.path.join(filedir, filename)
        np.save(filename, energy_list)
    else:
        filename = args.type_model + f'_layer{args.num_layers}_{c_min}_{c_max}_{loss_weight}_{bias_SReLU}.npy'
        filename = os.path.join(filedir, filename)
        np.save(filename, np.array(list_acc))


if __name__ == "__main__":
    # args = build_controller_args()
    args = BaseOptions().initialize()
    set_seed(args)
    main(args)