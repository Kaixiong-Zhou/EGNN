# Dirichlet Energy Constrained Learning for Deep Graph Neural Network

This is an authors' implementation of "Dirichlet Energy Constrained Learning for Deep Graph Neural Network" in Pytorch.

Authors: Anonymous Authors

This work is underreview by ICML 2021


## Introduction

This work presents a Dirichlet Energy Constrained Learning to guide the 
design and training of deep graph neurall networks.

Based on the principle, we propose a new model named  Energetic GraphNeural Networks (EGNN).
It could be easily constructed and trained to enable deep layer stacking. 


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


--type_model: the type of GNN model. We include ['GCN', 'pairnorm', 'EdgeDrop', 'simpleGCN'
'JKNet', 'APPNP', 'GCNII', 'EGNN']

--dataset: we include ['Cora', 'Pubmed', 'Coauthor_Physics', 'Ogbn-arxiv']






# EGNN
# EGNN
