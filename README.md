# Dirichlet Energy Constrained Learning for Deep Graph Neural Network

This is an authors' implementation of "Dirichlet Energy Constrained Learning for Deep Graph Neural Network" in Pytorch.

Authors: Kaixiong Zhou, Xiao Huang, Daochen Zha, Rui Chen, Li Li, Soo-Hyun Choi, Xia Hu

Paper: https://arxiv.org/abs/2107.02392

This work has been accepted by NeurIPS 2021


## Introduction

This work presents a Dirichlet Energy Constrained Learning to guide the 
design and training of deep graph neurall networks.

Based on the principle, we propose a new model named  Energetic GraphNeural Networks (EGNN).
It could be easily constructed and trained to enable deep layer stacking. 

The baseline GNN approaches and EGNN model accompanied with the default 
hyperparameters are incuded in this repository.

## Requirements

python == 3.6

torch == 1.7.1

torch-geometric==1.6.3

## Train EGNN model

To train EGNN run:
```
python main.py --cuda_num=0  --type_model='EGNN' --dataset='Cora' --num_layers=64
```
Hyperparameter explanations:


--type_model: the type of GNN model. We include ['GCN', 'pairnorm', 'EdgeDrop', 'SGC'
'JKNet', 'APPNP', 'GCNII', 'EGNN']

--dataset: we include ['Cora', 'Pubmed', 'Coauthor_Physics', 'ogbn-arxiv']

--num_layers: layers used in GNNs



## citation

If using this code, please cite our paper.
```
@article{zhou2021dirichlet,
  title={Dirichlet energy constrained learning for deep graph neural networks},
  author={Zhou, Kaixiong and Huang, Xiao and Zha, Daochen and Chen, Rui and Li, Li and Choi, Soo-Hyun and Hu, Xia},
  journal={Advances in neural information processing systems},
  year={2021}
}
```

