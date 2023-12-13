# Preprocessing of ChEMBL (2023.12.12)

1. Download ChEMBL all small molecule compounds (in **2023.12.12**, **1,920,643** compounds), save as **download_allcompds_1920643.tsv**
2. Prepare a script **init.py**, input **download_allcompds_1920643.tsv**, outputs **chembl_2023-12-12_train.txt** and **chembl_2023-12-12_valid.txt**

记录：
- 从 **download_allcompds_1920643.tsv** 中读入分子，共 1,920,643 个分子
- 去除 Smiles 为空（代码中 `type(s) != str`，因为通过 pandas 载入数据后，空值会以 Nan 表示，Nan 为 float 格式）后，剩余 1,919,184 个分子
- 去除混合物（代码中 Smiles 包含 `.` 的去除）后，剩余 1,814,044 个
- 去重（先 Kekulize 化，然后 Flatten，转化 InchiKey 后，取 InchiKey14 作为基准去重），得 1,626,027 个分子
- 使用 `random.shuffle(list)` 打乱后，取前 1,526,027 个分子存为 `chembl_2023-12-12_train.txt`，取后 100,000 个分子存为 `chembl_2023-12-12_valid.txt`，两个文件合并后存 `chembl_2023-12-12_unique.txt`

执行：

```
(dgl) $ python ../prepare_for_dgmg.py --tr_file chembl_2023-12-12_train.txt --val_file chembl_2023-12-12_valid.txt --out_types_file chembl_2023-12-12_ab_types.pkl
```

记录：
- 该脚本读入 train 和 valid 的 SMILES 文件，给出 atoms 和 bonds types
- 输出：原子为 7 个重原子（C, N, O, S, F, Cl, Br），键为单键、双键、三键

