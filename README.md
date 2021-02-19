# Dirichlet Energy Constrained Learning for Deep Graph Neural Network

This is an authors' implementation of "Benchmarking Deep Graph Neural Network" in Pytorch.

Authors: Anonymous Authors



## Introduction

Bag of tricks for deep GNN.

## Requirements

python == 3.6

torch == 1.7.1

torch-geometric==1.6.3

## Train GCN model

To train GCN run:
```
python main.py --cuda_num=0  --type_model='GCN' --dataset='Cora' --num_layers=64
```
Hyperparameter explanations:


--type_model: the type of GNN model. We include ['GCN', 'pairnorm', 'EdgeDrop', 'simpleGCN'
'JKNet', 'APPNP', 'GCNII']

--dataset: we include ['Cora', 'Pubmed', 'Coauthor_Physics', 'Ogbn-arxiv']






# EGNN
# EGNN
# Benchmark-deepGNN
