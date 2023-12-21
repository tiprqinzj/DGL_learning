import torch
import random, time, math
import numpy as np
from rdkit import Chem
from dgllife.model import DGMG

def cal_time(since):
    now = time.time()
    s = now - since
    
    if s > 86400:
        d = math.floor(s / 86400)
        h = math.floor((s - d * 86400) / 3600)
        m = math.floor((s - d * 86400 - h * 3600) / 60)
        s = s - d * 86400 - h * 3600 - m * 60
        out = '{}d {}h {}m {:.0f}s'.format(d, h, m, s)
    elif s > 3600:
        h = math.floor(s / 3600)
        m = math.floor((s - h * 3600) / 60)
        s = s - h * 3600 - m * 60
        out = '{}h {}m {:.0f}s'.format(h, m, s)
    else:
        m = math.floor(s / 60)
        s = s - m * 60
        out = '{}m {:.0f}s'.format(m, s)
    return out


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):

    torch.set_num_threads(1)
    set_random_seed(args['seed'])

    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
    bond_types = [Chem.rdchem.BondType.SINGLE,
                 Chem.rdchem.BondType.DOUBLE,
                 Chem.rdchem.BondType.TRIPLE]

    model = DGMG(atom_types=atom_types,
                 bond_types=bond_types,
                 node_hidden_size=128,
                 num_prop_rounds=2,
                 dropout=0.2)
   
    model.load_state_dict(torch.load(args['model_path'])['model_state_dict'])

    model.eval()

    smiles = []
    losses = []

    t_start = time.time()

    for i in range(args['num_mols']):
        with torch.no_grad():
            log_prob, s = model(rdkit_mol=True, compute_log_prob=True, max_num_steps=400)
            smiles.append(s)
            losses.append(- log_prob)

            if (i+1) % 100 == 0:
                print('generate {} mols, time {}'.format(i+1, cal_time(t_start)))

    valid_smiles = []
    valid_losses = []

    for s, l in zip(smiles, losses):
        if Chem.MolFromSmiles(s):
            valid_smiles.append(s)
            valid_losses.append(l)
    
    print('Valid {:.2%}, averaged loss {:.2f}'.format(
        len(valid_smiles)/len(smiles), np.mean(valid_losses)
    ))

    with open(args['out_file'], 'w') as f:
        for l, s in zip(valid_losses, valid_smiles):
            f.write('{:.2f}\t{}\n'.format(l, s))


if __name__ == '__main__':

    args = {
        'seed': 42,
        'model_path': 'canonical2_2023-12-18/save_epoch20.pth',
        'out_file': 'canonical2_2023-12-18/samples_epoch20_seed42.smi',
        'num_mols': 500,
    }

    main(args)
