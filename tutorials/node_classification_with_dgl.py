# learn the basic workflow of using GNNs for node classification (predict the category of a node in a graph)

import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']

    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute Loss (only losses of nodes in the training set)
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # save the best validation accuracy
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print('In epoch {}, loss: {:.3f}, val acc: {:.3f} (best {:.3f}), test acc: {:.3f} (best {:.3f})'\
                .format(e, loss, val_acc, best_val_acc, test_acc, best_test_acc))



if __name__ == '__main__':

    # Loading Cora Dataset
    dataset = dgl.data.CoraGraphDataset()
    print(f'Number of categories: {dataset.num_classes}')

    # The task is to predict the category of a given paper.
    # Each paper

    # A DGL Dataset object may contain one or multiple graphs.
    # The Cora dataset used in this tutorial only consists of one single graph.
    g = dataset[0]

    # A DGL graph can store node features and edge features in two dictionary-like attributes called ndata and edata.
    print("Node features", g.ndata)
    print("Edge features", g.edata)

    print(g.ndata['feat'].shape)

    # Defining a Graph Convolutional Network (GCN)
    # a two-layer GCN, each layer computes new node representations by aggregating neighbor information
    # create the model with given dimensions
    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)

    # Training the GCN
    train(g, model)

    # Training on GPU, require to put both the model and the graph onto GPU
    # g = g.to('cuda')
    # model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to('cuda')
    # train(g, model)
