import random, time, math
import torch
import numpy as np
from rdkit import Chem
from dgllife.model.model_zoo.dgmg import MoleculeEnv
from torch.utils.data import Dataset, DataLoader
from dgllife.model import DGMG
import torch.distributed as dist
from datetime import timedelta

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

def load_smiles_from_file(fname):
    with open(fname) as f:
        smiles = f.read().splitlines()
    return smiles

class MoleculeDataset(object):
    '''
    order: random or canonical
    '''
    def __init__(self, in_file, order='random', subset_id=0, n_subsets=1):
        super(MoleculeDataset, self).__init__()

        self.in_file = in_file
        self.order = order
        self.subset_id = subset_id
        self.n_subsets = n_subsets
        self._setup()

    def collate(self, samples):
        assert len(samples) == 1
        return samples[0]

    def _create_a_subset(self, smiles):
        # e.g., 5 / 2 -> 2.5, 5 // 2 -> 2
        subset_size = len(smiles) // self.n_subsets
        return Subset(smiles[self.subset_id * subset_size: (self.subset_id + 1) * subset_size],
                      self.order, self.env)

    def _setup(self):
        self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
        self.bond_types = [Chem.rdchem.BondType.SINGLE,
                           Chem.rdchem.BondType.DOUBLE,
                           Chem.rdchem.BondType.TRIPLE]
        smiles = load_smiles_from_file(self.in_file)
        
        self.env = MoleculeEnv(self.atom_types, self.bond_types)
        self.data_set = self._create_a_subset(smiles)


class Subset(Dataset):
    def __init__(self, smiles, order, env):
        super(Subset, self).__init__()
        self.smiles = smiles
        self.order = order
        self.env = env
        self._setup()

    def _setup(self):
        mols = []
        for s in self.smiles:
            m = Chem.MolFromSmiles(s)
            Chem.Kekulize(m, clearAromaticFlags=True)
            mols.append(m)
        self.mols = mols

        if self.order == 'random':
            return

        self.decisions = []
        for m in self.mols:
            self.decisions.append(
                self.env.get_decision_sequence(m, list(range(m.GetNumAtoms())))
            )

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, item):
        if self.order == 'canonical':
            return self.decisions[item]
        else:
            m = self.mols[item]
            nodes = list(range(m.GetNumAtoms()))
            random.shuffle(nodes)
            return self.env.get_decision_sequence(m, nodes)


def cal_loss(rank, model, data_loader):
    model.eval()
    loss_ls = []
    t = time.time()

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            log_prob = model(actions=data, compute_log_prob=True).detach()
            loss = - log_prob
            loss_ls.append(loss.item())

            if ((i + 1) % 1000 == 0) and (rank == 0):
                print('process {} / {}, time {}'.format(i+1, len(data_loader), cal_time(t)))
        
    return loss_ls


def main(rank, args):

    time_start = time.time()
    
    set_random_seed(42)
    torch.set_num_threads(1)

    dataset = MoleculeDataset(args['in_file'], order=args['order'], subset_id=rank, n_subsets=args['num_processes'])
    
    # shuffle == False
    data_loader = DataLoader(dataset.data_set, batch_size=1, shuffle=False, collate_fn=dataset.collate)

    # initialize model
    model = DGMG(atom_types = dataset.atom_types,
                 bond_types = dataset.bond_types,
                 node_hidden_size = args['node_hidden_size'],
                 num_prop_rounds = args['num_prop_rounds'],
                 dropout = args['dropout'])
    
    # load model
    model.load_state_dict(torch.load(args['model_file'])['model_state_dict'])
    
    # cal loss
    loss_ls = cal_loss(rank, model, data_loader)
    
    # smiles
    smiles = dataset.data_set.smiles

    # save loss to file
    with open('{}/rank{}_loss.dat'.format(args['out_dir'], rank), 'w') as f:
        for s, l in zip(smiles, loss_ls):
            f.write('{}\t{:.2f}\n'.format(s, l))

    if rank == 0:
        print('Done, time {}'.format(cal_time(time_start)))


def launch_a_process(rank, args, target, minutes=7200):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args['master_ip'], master_port=args['master_port'])
    dist.init_process_group(backend='gloo',
                            init_method=dist_init_method,
                            timeout=timedelta(minutes=minutes),
                            world_size=args['num_processes'],
                            rank=rank)
    assert torch.distributed.get_rank() == rank
    target(rank, args)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Calculate losses for a given dataset')

    parser.add_argument('--model_file', type=str, metavar='', help='model PTH file')
    parser.add_argument('--in_file', type=str, metavar='', help='a TXT file that will be used for calculation')
    parser.add_argument('--out_dir', type=str, metavar='', help='output dir')
    parser.add_argument('--order', type=str, choices=['random', 'canonical'], metavar='', default='random', help='random or canonical')
    parser.add_argument('--node_hidden_size', type=int, metavar='', default=128, help='default 128')
    parser.add_argument('--num_propagation_rounds', type=int, metavar='', default=2, help='default 2')
    parser.add_argument('--dropout', type=float, metavar='', default=0.2, help='default 0.2')
    parser.add_argument('--num_processes', type=int, metavar='', default=32, help='default 32')
    parser.add_argument('--master_ip', type=str, metavar='', default='127.0.0.1', help='default 127.0.0.1')
    parser.add_argument('--master_port', type=str, metavar='', default='12345', help='default 12345')

    # unpacked args
    args = parser.parse_args().__dict__

    # args = {
    #     'model_file':  'save_epoch10.pth',
    #     'in_file': '../../ChEMBL_DGMG_val.txt',
    #     'out_dir': 'cal_loss_for_tutorial_val_mols_by_epoch10',
    #     'num_processes': 32,
    #     'master_ip': '127.0.0.1',
    #     'master_port': '12345'
    # }

    mp = torch.multiprocessing.get_context('spawn')
    procs = []
    for rank in range(args['num_processes']):
        procs.append(mp.Process(target=launch_a_process, args=(rank, args, main), daemon=True))
        procs[-1].start()
    for p in procs:
        p.join()
    
