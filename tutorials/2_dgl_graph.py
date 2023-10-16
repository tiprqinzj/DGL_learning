import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import numpy as np
import torch


if __name__ == '__main__':

    # the below two statements are same
    g = dgl.graph(([0, 0, 0, 0, 0], [1, 2, 3, 4, 5]), num_nodes=6)
    g = dgl.graph((torch.LongTensor([0, 0, 0, 0, 0]), torch.LongTensor([1, 2, 3, 4, 5])), num_nodes=6)

    print(g.edges())


    # Assigning Node and Edge Features to Graph
    # assign a 3-dimensional node features vector for each node
    g.ndata['x'] = torch.randn(6, 3)
    # assign a 4-dimensional edge features vector for each edge
    g.edata['a'] = torch.randn(5, 4)
    # assign a 5x4 node feature matrix for each node
    g.ndata['y'] = torch.randn(6, 5, 4)

    print(g.edata['a'])


    # Querying Graph Structures
    print(g.num_nodes())
    print(g.num_edges())
    print(g.out_degrees(0))
    print(g.in_degrees(0))

    # Graph Transformations
    sg1 = g.subgraph([0, 1, 3])
    sg2 = g.edge_subgraph([0, 1, 3])

    print(sg1.ndata[dgl.NID]) # the original IDs of each node in sg1
    print(sg1.edata[dgl.EID]) # the original IDs of each edge in sg1
    print(sg2.ndata[dgl.NID]) # the original IDs of each node in sg2
    print(sg2.edata[dgl.EID]) # the original IDs of each node in sg2

    # subgraph and edge_subgraph also copies the original features to the subgraph
    print(sg1.ndata['x'])
    print(sg1.edata['a'])
    print(sg2.ndata['x'])
    print(sg2.edata['a'])

    # add a reverse edge for each edge in the original graph
    newg = dgl.add_reverse_edges(g)
    print(newg.edges())

    # Loading and Saving Graphs
    # save graphs
    dgl.save_graphs('data/graph.dgl', g)
    dgl.save_graphs('data/graphs.dgl', [g, sg1, sg2])

    # load graphs
    (g,), _ = dgl.load_graphs('data/graph.dgl')
    print(g)
    print(_)

    (g, sg1, sg2), _ = dgl.load_graphs('data/graphs.dgl')
    print(g)
    print(sg1)
    print(sg2)
