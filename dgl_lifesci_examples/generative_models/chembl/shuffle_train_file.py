import numpy as np

def load_smiles_from_file(fname):
    with open(fname) as f:
        smiles = f.read().splitlines()
    return smiles


def main(in_file, out_file, seed):

    # load input file
    smiles = load_smiles_from_file(in_file)

    # shuffle
    np.random.seed(seed)
    np.random.shuffle(smiles)

    # save
    with open(out_file, 'w') as f:
        for s in smiles:
            f.write(s + '\n')

if __name__ == '__main__':

    for i in range(50):
        main(
            in_file='chembl_2023-12-18_dgmg_train.txt',
            out_file='canonical3_2023-12-21/train_for_epoch{}.txt'.format(i+1),
            seed=i+1
        )
