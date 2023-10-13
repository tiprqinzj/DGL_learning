# Learning Notes of DGL

## Conda environment installation

 - date: 2023.10.13
 - the latest Python: 3.12.0

### Create Conda Environment

two machines: MacBook (my labtop) and HPC181 (company's)

```
# create date: 2023.10.13
(base) $ conda create -n dgl python=3.9 # 3.9.18 installed
(base) $ conda activate dgl

# For MacBook
(dgl) $ conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
(dgl) $ conda install -c dglteam dgl

# For HPC181
(dgl) $ conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
(dgl) $ conda install -c dglteam/label/cu116 dgl
(dgl) $ pip install packaging

(dgl) $ conda install -c conda-forge rdkit rdkit
```

*Note*: when execute `import dgl` in HPC181, an error occurred (No module named 'packaging'); `pip install packaging` deal with it.

## Tutorial 1: Node Classification with DGL

 - offcial tutorial: https://docs.dgl.ai/tutorials/blitz/1_introduction.html
 - create a folder named **tutorials** and cd to it
 - create a script named **node_classification_with_dgl.py**

This example is a node classification problem as a semi-supervised node classification task. Only a small portion of labeled nodes, a graph neural network (GNN) can accurately predict the node category of the others.  

## Tutorial 2: How does DGL represent a graph?

 - create a script named **how_does_dgl_represent_a_graph.py**

DGL represents a directed graph as a `DGLGraph` object. You can construct a graph by specifying the number of nodes in the graph as well as the list of *source and destination nodes*. Nodes in the graph have consecutive IDs starting from 0.

*Note*: For `DGLGraph`, the messages sent form one node to the other are often different both directions. If you want to handle *undirected graphs*, you may consider treating it as a bidirectional graph. 

Assigning Node and Edge Features to Graph
`DGLGraph` only accepts attributes stored in tensors (with numerical contents); an attribute of all the nodes and edges **must** have the same shape.

*Note*: encode various types of attributes into numerical features, some suggestions:
 - For categorical attributes, consider converting them to integers or one-hot encoding
 - For variable length string contents, consider applying a language model
 - For images, consider applying a vision model such as CNNs

### Querying Graph Structures

 - `g.num_nodes()`
 - `g.num_edges()`
 - `g.out_degrees(0)`: out degrees of the center node
 - `g.in_degrees(0)`: in degrees of the center node, for directed graph this will be 0

