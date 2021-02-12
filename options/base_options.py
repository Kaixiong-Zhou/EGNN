import argparse

def reset_weight(args):

    if args.dataset == 'Cora':
        args.loss_weight = 20. # 20. optimal
        args.c_min = 0.2 if args.num_layers < 32 else 0.15 # optimal 0.2/0.15
        args.alpha = 0.1 # optimal 0.1
        args.c_max = 1. # optimal 1.
        args.output_dropout = 0.6 #
        args.bias_SReLU = -10. # optimal -10
        if args.num_layers == 64:
            args.epochs = 1000

    elif args.dataset == 'Pubmed':
        args.loss_weight = 20.  # 20. optimal
        args.c_min = 0.12 if args.num_layers < 32 else 0.11 # 0.12、0.11
        args.alpha = 0.12 if args.num_layers < 32 else 0.11 # 0.12、0.11
        args.c_max = 1. # optimal 1.
        args.output_dropout = 0.
        args.bias_SReLU = -10. # optimal -10

    elif args.dataset == 'ogbn-arxiv':
        args.loss_weight = 1e-4 # 1e-4
        args.c_min = 0.6 if args.num_layers < 32 else 0.75 # 0.6, 0.75
        args.alpha = 0.1 if args.num_layers < 32 else 0.25 # 0.1, 0.25
        args.c_max = 1.
        args.output_dropout = 0.  # 0.
        args.bias_SReLU = -1 # optimal -1

    elif args.dataset == 'Coauthor_Physics':
        args.loss_weight = 20. # optimal 20.
        args.c_min = 0.12 # 0.12
        args.alpha = 0.1 # 0.1
        args.c_max = 1. # optimal 1.
        args.output_dropout = 0.6 # 0.6
        args.bias_SReLU = -5 # -5

    return args

class BaseOptions():

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""

    def initialize(self):
        parser = argparse.ArgumentParser(description='Constrained learing')

        # build up the common parameter
        parser.add_argument('--random_seed', type=int, default=100)
        parser.add_argument("--cuda", type=bool, default=True, required=False,
                            help="run in cuda mode")
        parser.add_argument('--cuda_num', type=int, default=0, help="GPU number")
        parser.add_argument("--dataset", type=str, default="Cora", required=False,
                            help="The input dataset.")

        # build up the supernet hyperparameter
        parser.add_argument('--type_model', type=str, default="EGNN")
        parser.add_argument('--num_layers', type=int, default=64)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument("--epochs", type=int, default=1500,
                            help="number of training the one shot model")
        parser.add_argument("--multi_label", type=bool, default=False,
                            help="multi_label or single_label task")
        parser.add_argument("--dropout", type=float, default=0.6,
                            help="input feature dropout")
        parser.add_argument("--lr", type=float, default=0.005,
                            help="learning rate")
        parser.add_argument('--dim_hidden', type=int, default=64)
        parser.add_argument('--transductive', type=bool, default=True)
        parser.add_argument('--activation',  type=str, default="relu", required=False)

        # mutual information
        parser.add_argument('--weight_decay1', type=float, default=0.,
                            help="decay for hidden GNN weight")
        parser.add_argument('--weight_decay2', type=float, default=5e-4,
                            help="decay for input/out transformation")  # 5e-4
        parser.add_argument('--weight_decay3', type=float, default=1e-4,
                            help="decay for SReLU shift")  # 5e-4
        parser.add_argument('--loss_weight', type=float, default=5.,
                            help="weight of orthogonal constraint")
        parser.add_argument('--bias_SReLU', type=float, default=-10,
                            help="initial bias in SReLU")
        parser.add_argument('--c_min', type=float, default=0.12,
                            help="lower bound of constrained learning")
        parser.add_argument('--alpha', type=float, default=0.1,
                            help="residual weight for input embedding")
        parser.add_argument('--c_max', type=float, default=1.,
                            help="upper bound of constrained learning")
        parser.add_argument('--output_dropout', type=float, default=0.2,
                            help="dropout rate at the final layer")
        parser.add_argument('--lamda', type=float, default=0.5,
                            help="used in GCNII")
        parser.add_argument('--wd1', type=float, default=0.01,
                            help="decay for GCNII")
        parser.add_argument('--wd2', type=float, default=5e-4,
                            help="decay for GCNII")  # 5e-4
        parser.add_argument('--patience', type=int, default=100,
                            help="patience step for early stopping")  # 5e-4
        parser.add_argument('--type_norm', type=str, default="None")
        parser.add_argument('--adj_dropout', type=float, default=0.5,
                            help="dropout rate in APPNP")  # 5e-4
        parser.add_argument('--edge_dropout', type=float, default=0.2,
                            help="dropout rate in EdgeDrop")  # 5e-4

        parser.add_argument('--compute_energy',
                            action='store_true', help="compute middle energy or not")  # 5e-4
        args = parser.parse_args()
        args = self.reset_model_parameter(args)
        return args

    def reset_model_parameter(self, args):


        if args.dataset == 'Cora':
            args.num_feats = 1433
            args.num_classes = 7
            args.dropout = 0.6 # 0.5
            args.lr = 0.005 # 0.005
            args.edge_dropout = args.edge_dropout if args.num_layers < 16 else 0.8
            args.adj_dropout = args.adj_dropout if args.num_layers < 16 else 0.8

            args.weight_decay2 = 0.


        elif args.dataset == 'Pubmed':
            args.num_feats = 500
            args.num_classes = 3
            args.hidden_dim = 256
            args.dropout = 0.5
            args.lr = 0.01
            args.lamda = 0.4 # For GCNII
            args.wd1 = 5e-4 # For GCNII
            args.adj_dropout = args.adj_dropout if args.num_layers < 16 else 0.9
            if args.num_layers == 2:
                args.edge_dropout = args.edge_dropout
            elif args.num_layers == 16:
                args.edge_dropout = 0.3
            elif  args.num_layers == 64:
                args.edge_dropout = 0.9


        elif args.dataset == 'Coauthor_Physics':
            args.num_feats = 8415
            args.num_classes = 5
            args.dropout = 0.6
            args.weight_decay2 = 5e-5
            args.wd2 = 5e-5  # for GCNII
            # for GCN based methods
            args.adj_dropout = 0.05 if args.num_layers < 16 else 0.9
            args.edge_dropout = args.edge_dropout if args.num_layers <= 16 else 0.9

        elif args.dataset == 'ogbn-arxiv':
            args.num_feats = 128
            args.num_classes = 40
            args.hidden_dim = 256
            args.dropout = 0.1
            args.lr = 0.003
            args.epochs = 1000
            args.patience = 200
            args.alpha = 0.5
            args.weight_decay2 = 0. # for EGNN
            args.weight_decay3 = 0.
            args.wd1 = args.wd2 = 0. # for GCNII

            # for GCN based methods
            args.type_norm = 'pair' if args.type_model == 'pairnorm' else 'batch'
            args.adj_dropout = 0.05
            args.edge_dropout = 0.1

        return args