# DGL-Lifesci 案例学习笔记

- 来源：https://github.com/tiprqinzj/dgl-lifesci
- 源码下载日期：2023-11-09
- chuangxin06 保存位置：D:\github\dgl-lifesci-master
- MacBook 保存位置：/Users/qinzijian/Project/dgl-lifesci-master

## 一、性质预测

新建文件夹：**property_prediction**

### 源码笔记学习

MoleculeNet 是一组较为流行的机器学习基准分子集，源码中作者使用了 12 个数据集验证 DGL-LifeSci 的效果，其中有 9 个分类数据集（BACE, BBBP, ClinTox, HIV, MUV, PCBA, SIDER, Tox21, ToxCast）和 3 个回归数据集（ESOL, FreeSolv, Lipophilicity）

对于每个数据集，使用分层划分，并使用贝叶斯优化器执行超参数搜索（共 32 组），对于每组测试，训练一个随机初始化的模型，在一定轮数后训练直到验证集的效果不再提升；在 32 组测试中验证集最佳的模型用于评价测试集

训练一个分类模型，执行如下命令：

```
python classification.py -d DATASET -mo MODEL -f FEATURE
```

其中：
- `DATASET` 指定数据集，可以为以下 9 个中的任意一个 `MUV, BACE, BBBP, ClinTox, SIDER, ToxCast, HIV, PCBA, Tox21`
- `MODEL` 指定模型，可以为以下 10 个中的任意一个 `GCN, GAT, Weave, MPNN, AttentiveFP, gin_supercised_contextpred, gin_supervised_infomax, gin_supervised_edgepred, gin_supervised_masking, NF`
- `FEATURE` 指定特征化方法，可以为以下 2 个中的任意一个 `canonical, attentivefp`，如果模型方法为 `gin_supervised_*` 则可忽略该参数

### 源码脚本学习

复制以下源码中的文件至当前文件夹：
- examples/property_prediction/moleculenet/classification.py
- examples/property_prediction/moleculenet/utils.py
- examples/property_prediction/moleculenet/configures/*

执行以下命令：

```
python classification.py -d BBBP -mo GCN -f canonical`
```

成功执行，默认训练 1000 轮，在第 98 轮训练后 EarlyStop，测试集 AUC 为 0.6364

创建 Jupyter 脚本，将代码拆解学习，创建 classification.ipynb


### 源码学习笔记汇总：对 BBBP 数据使用 GCN 模型创建二分类模型

（一）导入需要的包和函数

```
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import dgl
from dgllife.data import BBBP
from dgllife.model import GCNPredictor
from dgllife.utils import EarlyStopping, Meter, SMILESToBigraph
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import ScaffoldSplitter
```

（二）建模所需输入参数

```
args = {
    'num_epochs': 1000,
    'num_workers': 0,
    'print_every': 10,
    'result_path': 'classification_results',
    'device': torch.device('cpu'),
    'node_featurizer': CanonicalAtomFeaturizer(),
    'edge_featurizer': None,
}
```

（三）自定义 SMILES To Graph 函数

```
smiles_to_g = SMILESToBigraph(add_self_loop=True,
                              node_featurizer=args['node_featurizer'],
                              edge_featurizer=args['edge_featurizer'])
```

注：
- `add_self_loop` 设置为 `False` 会导致训练出错
- `CanonicalAtomFeaturizer` 将原子性质转化为 74 位的 one-hot 编码，特征名为`'h'`，其中有不少原子类别其实是不存在的，如果有需要可以在后续时做相应切片删除
- `CanonicalAtomFeaturizer().feat_size()` 将返回 74

（四）载入数据库

```
dataset = BBBP(smiles_to_graph=smiles_to_g, n_jobs=1)

>>> print(len(dataset))
2039

>>> print(dataset[0]) # a tuple with 4 items, that is (SMILES, DGLGraph, Label Tensor, multi-label mask Tensor)
```

（五）划分训练集与测试集

```
train_set, val_set, test_set = ScaffoldSplitter.train_val_test_split(
    dataset, frac_train=0.8, frac_val=0.1, frac_test=0.1
)
```

查看 `dataset` 和 `train_set` 等的数据类型：
  - `dataset`: `<dglife.data.bbbp.BBBP>`
  - `train_set`: `<dgl.data.utils.Subset>`

（六）DataLoader

```
def collate_molgraphs(data):
    # uppacked the dataset to list
    smiles, graphs, labels, masks = map(list, zip(*data))

    # batch graphs
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, labels, masks

train_loader = DataLoader(dataset=train_set,
                          batch_size=64, # config['batch_size']
                          shuffle=True,
                          collate_fn=collate_molgraphs,
                          num_workers=0, # args['num_workers']
                         )
# similar of val_loader and test_loader
```

记录：
- collate：整理
- 整理函数读入一个 dataset（或者一个 subset），函数将其数据解包，并使用 `dgl.batch()`` 方法返回每批的数据

（七）准备 GCN 模型及模型参数

```
config = {
    'batch_size': 64,
    'batchnorm': False,
    'dropout': 0.025,
    'gnn_hidden_feats': 256,
    'lr': 0.02,
    'num_gnn_layers': 4,
    'patience': 30,
    'predictor_hidden_feats': 32,
    'residual': True,
    'weight_decay': 0.001,
    'in_node_feats': 74,
    'n_tasks': 1,
}

model = GCNPredictor(in_feats = 74, # config['in_node_feats']
                     hidden_features = [256, 256, 256, 256], # [config['gnn_hidden_feats']] * config['num_gnn_layers]
                     activation = [F.relu, F.relu, F.relu, F.relu], # [F.relu] * config['num_gnn_layers']
                     residual = [True, True, True, True], # [config['residual']] * config['num_gnn_layers']
                     batchnorm = [False, False, False, False], # [config['batchnorm']] * config['num_gnn_layers']
                     dropout = [0.025, 0.025, 0.025, 0.025], # [config['dropout']] * config['num_gnn_layers']
                     predictor_hidden_feats = 32,# config['predictor_hidden_feats']
                     predictor_dropout = 0.025, # config['dropout']
                     n_tasks = 1, # config['n_tasks']
                    )
model = model.to(args['device'])
```

`GCNPredictor` 参数解读：
- `in_feats`: int, 节点的输入特征数
- `hidden_features`: list of int, 例如 [64, 64] 代表有两个 GCN 隐含层，每层均为 64 个节点
- `gnn_norm`: list of str, 每个 GCN 层的消息传递标准化方法，可以为 `'right'`, `'both'`, `'none'`，其中 `'none'` 表示简单将信息加和，默认为 `['none', 'none']`
- `activation`: list of activation functions or None, 每层的激活函数
- `residual`: list of bool, 是否使用 residual connection，默认使用 **没看懂**
- `batchnorm`: list of bool, 是否对 GCN 的输出值执行批的标准化，默认使用，源码中改为不使用 **没看懂**
- `dropout`: list of float, 每个 GCN 层的 dropout 比例，默认为无
- `predictor_hidden_feats`: int, 输出 MLP 隐含层的节点数
- `predictor_dropout`: float, 输出 MLP 隐含层的 dropout 比例
- `n_tasks`: int, 任务数，也是输出的大小

如何使用模型得到预测值：
- 按照源码方法，首先将 bg 的节点特征 pop 后存为 feats Tensor，然后将 bg 和 feats Tensor 输入到模型中
- `GCNPredictor`

（八）准备损失函数、优化器和 EarlyStop 追踪器

```
loss_criterion = nn.BCEWithLogitsLoss(reduction='none')
optimizer = Adam(model.parameters(), lr=0.02, weight_decay=0.001) # config['lr'], config['weight_decay']
stopper = EarlyStopping(patience=30, filename=args['result_path']+'/model.pth', metric='roc_auc_score')
```

`nn.BCEWithLogitsLoss` 函数：
- 将 `Sigmoid` 和 `BCELoss` 组合在一层，函数的说明文档表示，这比先 Sigmoid 再接一个 BCELoss 要更加稳定
- unreduced loss（即 `reduction` 设置为 `'none'`）可以描述为以下公式
- $\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right]$
- 其中 N 为 batchsize

`EarlyStopping`：
- 官方文档：当获得对验证集表现更好的模型时保存模型检查点，并在没有观测到验证集提升后的一定轮数后停止训练
- 英文：Save model checkpoint when observing a performance improvement on the validation set and early stop if improvement has not been observed for a particular number of epochs.
- 四个参数：`mode`, `patience`, `filename`, `metric`
  - `mode`: `higher`（默认） or `lower`
  - `patience`: 在该数值的轮数还未获得对验证集更好的模型时，就停止训练
  - `filename`: 保存 checkpoint 模型的文件名
  - `metric`: 从 `r2`, `mae`, `rmse`, `roc_auc_score` 中选择其一
- 该函数最终返回 bool 变量，当返回 True 时，停止训练

（九）准备训练、验证、预测函数

源码作者准备了三个函数，之后 Torch 神经网络训练可以采纳这种三函数策略：
- `run_a_train_epoch(args, epoch, model, data_loader, loss_criterion, optimizer)`: no returns
- `run_an_eval_epoch(args, model, data_loader)`: return metric
- `predict(args, model, bg)`: return logits, 在上述两个函数中调用

在源码中，还有一个神奇的函数 `Meter()`，这是一个方便追踪模型效果的函数，但个人认为只有 AUC 值还是不够，后续正式模型中还是需要写自己的模型评价函数，在此处暂时先用该函数。

`predict()` 函数示例：
- 其中的`pop`方法，将特征数据从原`bg`中删除了，但由于bg是通过`dataloader`遍历来的，只用这一次，所以不影响其它数据
- 本例中只有节点特征，若需要边的特征，则按需求增加即可
- 返回的 logits 其实就是 score 值

```
def predict(model, bg, device):
    bg = bg.to(device)
    node_feats = bg.ndata.pop('h').to(device)
    return model(bg, node_feats)
```

`run_a_train_epoch` 示例：
- 需要在最开始写明是用于训练过程，即`model.train()`
- 对数据进行解包，至少需要 `bg` 和 `labels`，将 `labels` 传递至计算设备（由于最开始 model 已经使用过 `.to()` 方法，`predict()` 函数中再将 bg 传递至计算设备）
- 使用 `perdict()` 函数获取 logits 值
- 计算 loss
- 三句命令执行参数调整

```
def run_a_train_epoch(model, data_loader, loss_criterion, optimizer, device):
    model.train()

    for batch_id, batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels, masks = labels.to(device), masks.to(device)
        
        logits = predict(model, bg, device)
        loss = (loss_criterion(logits, labels) * (masks != 0).float()).mean()
        # loss.item() to obtain loss value

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # print something
```

`run_an_eval_epoch()` 示例：
- 需要在最前面写明为验证过程 `model.eval()`
- 在无需追踪梯度计算的代码块中执行：`with torch.no_grad()`

```
def run_an_eval_epoch(model, data_loader, device):
    model.eval()
    with torch.no_grad():
        for batch_data in enumerate(data_loader):
        smiles, bg, labels, masks = batch_data
        labels = labels.to(device)
        logits = predict(model, bg, device)
        # get loss
        # print something
    # return something 
```

### 后续需要学习的内容

- 如何创建自己的 DataLoader
- 将预测的 logits 转化为 score，并计算其它 metrics

## 二、创建自己的数据集

### 源码参考

从以下脚本中找到相关内容学习：
- examples/property_prediction/csv_data_configuration/classification_train.py
  - `load_dataset()`
- examples/property_prediction/csv_data_configuration/utils.py

### 源码学习

- 创建 `custom_dataset.ipynb` 拆解学习
- 以 Caco-2 AB 的数据集 *curated1805_trtecv.csv* 作为案例，存于 **data/caco2ab_curated1805_trtecv.csv**

### 学习笔记汇总：从 CSV 创建基于图的 dataset

（一）导入需要的包

```
import pandas as pd
import dgl
import torch
from torch.utils.data import DataLoader
from dgllife.data import MoleculeCSVDataset
from dgllife.utils import SMILESToBigraph
from dgllife.utils import CanonicalAtomFeaturizer
```

（二）准备整理函数，与性质预测中相同

- 需注意，`dgllife` 自带函数中就有 masks 这个参数，因此即使无空值也需要保留该参数

```
def collate_molgraphs(data):
    # uppacked the dataset to list
    smiles, graphs, labels, masks = map(list, zip(*data))

    # batch graphs
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(labels, dim=0)

    return smiles, bg, labels, masks
```

（三）从 `DataFrame` 载入数据库

自定义 `load_dataset` 函数，详解 `MoleculeCSVDataset` 函数参数用法：
- `smiles_column`: DataFrame 中，SMILES 所在列的列名
- `cache_file_path`: str, 将 SMILES 编码为 DGLGraph 后所保存的图文件路径，多数以 `.bin` 结尾
- `load`: bool, 默认为 False，是否载入本地 `cache_file_path` 文件并返回图变量，更改为 True 即可，因为观察源码后得知，只有当 `cache_file_path` 文件存在且 `load` 为 True 时，才会从本地文件载入
- `task_names`: list of str, labels 所在列的列名，保存为列表

```
def load_dataset(df, smi_title, label_title, out_graph_path):
    smiles_to_g = SMILESToBigraph(add_self_loop=True,
                                  node_featurizer=CanonicalAtomFeaturizer(),
                                  edge_featurizer=None)
    dataset = MoleculeCSVDataset(df=df,
                                 smiles_to_graph=smiles_to_g,
                                 smiles_column=smi_title,
                                 cache_file_path=out_graph_path,
                                 task_names=[label_title],
                                 load=True,
                                 n_jobs=1)
    return dataset

df = pd.read_csv(CSV_PATH)
dataset = load_dataset(df, SMI_TITLE, TARGET_TITLE, GRAPH_PATH)
```

（四）创建 DataLoader

```
data_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True,
                         collate_fn=collate_molgraphs, num_workers=0)

for data in dataloader:
    smiles, bg, labels, masks = data
    # do something
```

## 三、DGMG 生成模型

## Abstract of Source code and README

DGMG: Deep Generative Models of Graphs

**Goal**: Given a set of real molecules, we want to learn the distribution of them and get new molecules with similar properties.

### i) Dataset

With our implementation, this model has several limitations:
- Information about protonation and chirality are ignored during generation.
- Molecules consisting of `[N+]`, `[O-]`, etc. cannot be generated.

**ChEMBL**: at most 20 heavy atoms, and used a training/validation split of 130,830/26,166 molecules. 

**ZINC**: 232,464 for training and 5000 for validation.

### ii) Usage

Training auto-regressive generative models tends to be very slow. According to the authors, they use multiprocess to speed up training and GPU does not give much speed advantage.

