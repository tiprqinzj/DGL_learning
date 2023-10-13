# learn the basic workflow of using GNNs for node classification (predict the category of a node in a graph)

import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == '__main__':

    # Loading Cora Dataset
    dataset = dgl.data.CoraGraphDataset()
    print(f'Number of categories: {dataset.num_classes}')

    # The task is to predict the category of a given paper.

    # A DGL Dataset object may contain one or multiple graphs.
    # The Cora dataset used in this tutorial only consists of one single graph.
    g = dataset[0]

    # A DGL graph can store node features and edge features in two dictionary-like attributes called ndata and edata.
    
