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


