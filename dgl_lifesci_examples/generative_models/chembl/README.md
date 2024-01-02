# Preprocessing of ChEMBL (2023.12.12)

1. Download ChEMBL all small molecule compounds (in **2023.12.12**, **1,920,643** compounds), save as **download_allcompds_1920643.tsv**
2. Prepare a script **init.py**, input **download_allcompds_1920643.tsv**, outputs **chembl_2023-12-12_train.txt** and **chembl_2023-12-12_valid.txt**

记录：
- 从 **download_allcompds_1920643.tsv** 中读入分子，共 1,920,643 个分子
- 去除 Smiles 为空（代码中 `type(s) != str`，因为通过 pandas 载入数据后，空值会以 Nan 表示，Nan 为 float 格式）后，剩余 1,919,184 个分子
- 去除混合物（代码中 Smiles 包含 `.` 的去除）后，剩余 1,814,044 个
- 去重（先 Kekulize 化，然后 Flatten，转化 InchiKey 后，取 InchiKey14 作为基准去重），得 1,626,027 个分子
- 存 `chembl_2023-12-12_unique.txt`

执行：

```
(dgl) $ python ../prepare_for_dgmg.py --tr_file chembl_2023-12-12_train.txt --val_file chembl_2023-12-12_valid.txt --out_types_file chembl_2023-12-12_ab_types.pkl
```

记录：
- 该脚本读入 train 和 valid 的 SMILES 文件，给出 atoms 和 bonds types
- 输出：原子为 7 个重原子（C, N, O, S, F, Cl, Br），键为单键、双键、三键

新建 **get_dgmg_valid_smiles.py** 脚本，删除 unique.txt 中 DGMG 识别不了（false novel）的分子，执行：

```
(dgl) $ python get_dgmg_valid_smiles.py -i chembl_2023-12-12_unique.txt -o chembl_2023-12-12_dgmg.txt -e chembl_2023-12-12_invalid.txt
```

输出结果为 **chembl_2023-12-12_dgmg.txt**，包含 1,530,493 个分子；输出的 **chembl_2023-12-12_invalid.txt** 中为被过滤掉的 95535 个分子的信息。

将后 10 万分子存为验证集 **chembl_2023-12-12_dgmg_val.txt**，前 1,430,493 分子存为训练集 **chembl_2023-12-12_dgmg_train.txt**

准备训练脚本 **train.py**，该脚本借鉴自上一级文件夹的 **summary.py**，不同的是该脚本中标准化分子这一步，仅 Kekulize 即可，且无需再提取原子和键的类型，提前固定好（即原子为 C, N, O, S, F, Cl, Br，键为单键、双键和三键）。

```
(dgl) $ nohup python train.py --tr_file chembl_2023-12-12_dgmg_train.txt --val_file chembl_2023-12-12_dgmg_val.txt --log_file train.log --order canonical --save_prefix canonical > nohup_train_2023-12-14_1545pm.log 2>&1 &
```

需注意，nohup 后，即使 kill 掉 train.py 的进程，并行依然在运算，因此需要 ps -ef | grep spawn | cut -c 9-16 后获取全部 PID，在 NotePad++ 中将回车替换为空格，再 kill -9 PASTE 来彻底清空后台的进程，才可以继续提交后续计算任务。

**停止训练**：2023.12.18 8:00 am: 经过一个周末的训练后，valid set 损失出现 nan，训练集的 rank 14 出现 nan，该模型出现问题，终止训练。

查询可能出现 NaN 的原因：第一为学习率过高导致了梯度爆炸，第二为数据集本身的问题。

## 重整 ChEMBL 数据集

上一次失败的训练中，根据对训练损失的观察，有几个 rank 的损失明显比其它的要大，可能这几个 rank 中出现了比较奇怪的分子，导致学习出现了梯度非常大的情况，首先排查这些分子。

方法：使用保存下来的模型，将训练集和验证集都作为验证集，保存每个分子的 loss，观察哪些分子的 loss 很大，去手动核实这些分子是否比较奇怪，若较为奇怪删除即可。

在 **canonical** 文件夹内，准备 debug 脚本 **debug_cal_loss_for_allcompds.py**。使用训练 10 轮之后的模型运行，执行参数如下（1h 17m 完成计算）

```
args = {
    'model_file':  'save_epoch10.pth',
    'in_file': '../chembl_2023-12-12_dgmg.txt',
    'out_dir': 'cal_loss_for_all_by_epoch10',
    'num_processes': 100,
    'master_ip': '127.0.0.1',
    'master_port': '12345'
}
```

新建 **cal_loss_for_all_by_epoch25** 使用训练 25 轮之后的模型计算全部分子的损失（此时的 val loss 已出现 nan 情况），执行参数如下：

```
args = {
        'model_file':  'save_epoch10.pth',
        'in_file': '../chembl_2023-12-12_dgmg.txt',
        'out_dir': 'cal_loss_for_all_by_epoch25',
        'num_processes': 100,
        'master_ip': '127.0.0.1',
        'master_port': '12345'
    }
```

### 统计训练 10 轮后的损失

准备脚本 **combine_cal_loss_and_count.py**，其功能为将给定的 100 个 loss 数据合并后，计算 loss 的均值和方差，统计 loss 在不同范围内的数据量。输入参数和输出的屏幕打印如下：

```
# input
files_ls = ['cal_loss_for_all_by_epoch10/rank{}_loss.dat'.format(i) for i in range(100)]

# output
Totol 1530400 mols               
Loss average 41.50, STD 14.76
  loss in [0, 10):   0 mols
  loss in [10, 20):  23474 mols
  loss in [20, 30):  300962 mols
  loss in [30, 40):  502086 mols
  loss in [40, 50):  356045 mols
  loss in [50, 60):  188689 mols
  loss in [60, 70):  87508 mols
  loss in [70, 80):  39268 mols
  loss in [80, 90):  17293 mols
  loss in [90, 100): 7485 mols
  loss >= 100:       7590 mols
```

用训练 10 轮后的模型，评估 DGMG 教程中的验证集数据

```
# input of combine_cal_loss_and_count.py
files_ls = ['cal_loss_for_tutorial_val_mols_by_epoch10/rank{}_loss.dat'.format(i) for i in range(100)]

# output
Totol 130800 mols               
Loss average 30.42, STD 7.96    
  loss in [0, 10):   0 mols     
  loss in [10, 20):  8846 mols  
  loss in [20, 30):  59971 mols 
  loss in [30, 40):  46957 mols 
  loss in [40, 50):  12481 mols 
  loss in [50, 60):  2128 mols  
  loss in [60, 70):  354 mols   
  loss in [70, 80):  47 mols    
  loss in [80, 90):  11 mols    
  loss in [90, 100): 5 mols     
  loss >= 100:       0 mols     
```

### 提取新的训练数据集

根据训练 10 轮后，对全部 153 万分子评估的 loss 值，决定删除损失过大的分子。准备脚本 **pick_smiles_by_loss.py**

```
# inputs
files_ls = ['cal_loss_for_all_by_epoch10/rank{}_loss.dat'.format(i) for i in range(100)]

main(
    files_ls,
    loss_cutoff=85.78,
    out_file='../chembl_2023-12-18_dgmg.txt',
    filtered_file='../chembl_2023-12-18_filtered.txt'
)

# outputs
Passed mols 1509741, filtered mols 20659
```

提取出的新数据，保存在 **dgl_lifesci_examples/generative_models/chembl/chembl_2023-12-18_dgmg.txt** 中；被过滤掉的 20659 个分子，保存在 **dgl_lifesci_examples/generative_models/chembl/chembl_2023-12-18_filtered.txt** 中。

接下来，将使用新提取到的 1509741 分子进行训练。

## 重新训练，使用 2023-12-18 数据

将 **chembl_2023-12-18_dgmg.txt** 的后 10 万个分子拆分为验证集，其余为训练集，存于 **chembl_2023-12-18_dgmg_train.txt** 和 **chembl_2023-12-18_dgmg_val.txt**

训练脚本存副本 **train_backup2023-12-18.py**，修改原脚本，将原来每 epoch 后衰减学习率，改为每 1000 steps 后衰减学习率；修改，每个 rank 的 training loss 不再保存。


```
nohup python train.py --tr_file chembl_2023-12-18_dgmg_train.txt --val_file chembl_2023-12-18_dgmg_val.txt --log_file train.log --order canonical --save_prefix canonical2_2023-12-18 > nohup_train_2023-12-18_1530pm.log 2>&1 &
```

当运行完成 29 轮后，验证集的损失已基本收敛（约 40），此时学习率已衰减至 1e-6 以下，因此手动停掉训练。

## 修改训练思路 2023.12.21

已训练20轮，使用20轮后的模型生成500分子，valid 59.4%，平均loss为39.82。计划在下周一（12.25）再次训练一组模型，预计下周一在不崩溃的情况下可以训练50轮，到时学习率只有8e-8，也就是初始1e-4的千分之一不到，已经基本不调整权重了。预计50轮后验证集的损失也会在40左右，相对还是比较高。

第一次训练，学习率每轮衰减1%，导致到了22轮开始出现梯度爆炸；本次训练，学习率每1000步衰减1%，在16轮中（230次衰减），已降至初始学习率的十分之一，然而模型并未收敛，可能衰减过快了。本次训练拟计划每1500步衰减一次，每轮约衰减15次。

发现的问题：训练是拆分成100组，CPU并行训练，其实是每个CPU各自去使用自己的训练集去调整网络权重，最后会有一个将全部100组的参数求和后取平均的过程。这样就会有一个问题，每个CPU其实每次只是利用自己的1.4万分子训练，且顺序还是一致的；只有在全部100个CPU都训练完成后，这100个模型才互相通信，这就可能造成模型在自己数据集上的偏差。

调整思路：每个epoch后，将训练数据打乱，重新拆分成100份，创建dataloader，执行下一次训练。这样每轮训练时，模型学习的数据是有区别的，但都属于同一个大训练集。

创建 **train.py** 的副本 **train_update1.py**，在其中更改代码：
- 没有想到特别好的解决方案，于是决定提前shuffle好50轮的数据，存50个训练集smiles文件
- 准备脚本 **shuffle_train_file.py**，创建文件夹 **canonical3_2023-12-21**，将 **chembl_2023-12-18_dgmg_train.txt** 作为输入，输出50个文件 **canonical3_2023-12-21/train_for_epochX.txt (X = 1 ~ 50)**

```
nohup python train_update1.py --tr_file_prefix canonical3_2023-12-21/train --val_file chembl_2023-12-18_dgmg_val.txt --log_file train.log --order canonical --save_prefix canonical3_2023-12-21 > nohup_train_2023-12-22_1020am.log 2>&1 &
```

但在这次训练中，第一轮出现了上百万甚至千万损失的rank，需继续观察是否会对后续epoch产生影响。

## 发现严重训练错误（2023.12.22）

暂时终止 canonical3 的训练，创建 **canonical4_2023-12-22**

对脚本 **train.py** 浏览时，发现分布式计算过程的优化器代码中，竟然没有合并梯度！！！也就是虽然我表面上分成100组平行计算，但每次这100个计算完成后，只采纳了第1个rank的梯度进行模型参数调整！重新调整脚本后，执行以下代码运行。

```
nohup python train.py --tr_file chembl_2023-12-18_dgmg_train.txt --val_file chembl_2023-12-18_dgmg_val.txt --log_file train.log --order canonical --save_prefix canonical4_2023-12-22 > nohup_train_2023-12-22_1345pm.log 2>&1 &
```

运行后，每次 step，100 个损失值是同时写到 log 中的，这次对了（以前不同时写 log，没想通分布式的思路，这下子都明白了）。可以理解为，分布式的 num_processes，就是其它网络的 batch_size，只是 DGMG 模型自身的特性只能设置 batch_size 为 1，用分布式来模拟批次训练。

**2023.12.25 13:00pm 停止训练**，由于并行数量过高，导致机器温度报警，后续预计使用 64 或 32 核进行计算。停止时已完成 10 轮的计算，可以考虑断点续算，即读取10轮训练后的模型，更改成并行 32 后继续训练，而无需从头重新训练。

## 从之前训练10轮后的模型继续训练（2023.12.28）

创建 **train.py** 的副本，命名为 **train_for_continue.py**，在后者中更改。

创建文件夹 **canonical5_2023-12-28**，接续训练的数据都存在此处。学习率在上次训练的第 10 轮，12627 步为 0.0000392711。本次以 0.00004 为起始，由于从并行 100 减少至并行 32，相当于降低 3 倍多，因此调整原每 1500 步衰减学习率为每 5000 步。

```
nohup python train_for_continue.py --tr_file chembl_2023-12-18_dgmg_train.txt --val_file chembl_2023-12-18_dgmg_val.txt --log_file train.log --order canonical --pre_model_path canonical4_2023-12-22/save_epoch10.pth --pre_epochs 10 --lr 0.00004 --save_prefix canonical5_2023-12-28 > nohup_train_2023-12-28_1500pm.log 2>&1 &
```

2024.01.02 上班后，得知训练到 20 轮，损失仅降低了一点（从 33 降至 32），考虑继续训练 10 轮，如果依然没有明显变化，则之后重新训练，学习率衰减再慢一些。想到之前出现梯度爆炸的情况，可能是由于没有将并行任务合并，仅通过 rank 0 的一万多个分子调整权重，自然无法遍历全部的百万分子。

使用训练 20 轮之后的模型，评价源码中的验证集分子。准备 **cal_loss_for_compds.py**，并未计算，因为之前的任务还在训练 master port 等不清楚如何定义。

