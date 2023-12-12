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

# For MacBook (discontinued in 2023.10.16)
(dgl) $ conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
(dgl) $ conda install -c dglteam dgl
(dgl) $ conda install -c conda-forge rdkit rdkit
(dgl) $ conda install ipykernel nb_conda
(dgl) $ python -m ipykernel install --user --name dgl --display-name 'dgl'
(base) $ conda update nbconvert
(dgl) $ conda remove ipykernel nb_conda # this make the environment crash

# For MacBook (started in 2023.10.16)
(base) $ conda create -n dgl2 python=3.9
(dgl2) $ conda install jupyter notebook
(dgl2) $ conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 -c pytorch
(dgl2) $ conda install -c dglteam dgl
(dgl2) $ conda install -c conda-forge rdkit rdkit
(dgl2) $ conda install -c conda-forge scikit-learn # in 2023.10.17

# For MacBook (started in 2023.11.09), install dgl-lifesci
(dgl2) $ conda create --name dgl3 --clone dgl2
(dgl2) $ conda activate dgl3
(dgl3) $ pip install dgllife

# For HPC181
(dgl) $ conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.6 -c pytorch -c nvidia
(dgl) $ conda install -c dglteam/label/cu116 dgl
(dgl) $ pip install packaging

(dgl) $ conda install -c conda-forge rdkit rdkit
(dgl) $ conda install -c conda-forge scikit-learn # in 2023.12.12
(dgl) $ pip install dgllife # in 2023.12.12
```
*Note*: when execute `jupyter notebook` in MacBook, an error occurred (500: Internal Server Error), main reason is that the version of jupyter in base and dgl is different; but after update, still not solve the problem

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

### DGL Graph Constructions

An example:
```
g = dgl.graph([0,0,0,0,0], [1,2,3,4,5], num_nodes=6) # 6 nodes, 5 edges
g.ndata['x'] = torch.randn(6, 3) # 6 nodes, 3 features
g.edata['a'] = torch.randn(5, 4) # 5 edges, 4 features
g.ndata['y'] = torch.randn(6, 5, 4) # a 5x4 node feature matrix for each node
```


### Querying Graph Structures

 - `g.num_nodes()`
 - `g.num_edges()`
 - `g.out_degrees(0)`: out degrees of the center node
 - `g.in_degrees(0)`: in degrees of the center node, for directed graph this will be 0

### Graph Transformations

 - `g.subgraph()`
 - `g.edge_subgraph()`
 - `dgl.add_reverse_edges()`

*Note*: if you have an undirected graph, it is better to convert it into a bidirectional graph first via adding reverse edges.

### Loading and Saving Graphs

 - `dgl.save_graphs()`
 - `dgl.load_graphs()`



## User Guide

Working folder: **user_guide**  

The *ipynb* notebook in this folder is unfinished. Only ch1 is useful.

# 学习 DGL-lifesci 教程

- 工作文件夹：**dgl_lifesci_examples**
- 笔记跳转至 [此处](dgl_lifesci_examples/README.md)

