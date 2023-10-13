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

(dgl) $ conda install -c conda-forge rdkit rdkit
```

## Tutorial 1: Node Classification with DGL

 - offcial tutorial: https://docs.dgl.ai/tutorials/blitz/1_introduction.html
 - create a folder named **tutorials** and cd to it
 - create a script named **node_classification_with_dgl.py**


