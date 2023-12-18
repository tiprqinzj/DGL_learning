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