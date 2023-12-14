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

