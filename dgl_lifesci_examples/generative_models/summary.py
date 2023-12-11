import time, math, random
from rdkit import Chem
from rdkit.Chem import AllChem
from dgllife.model.model_zoo.dgmg import MoleculeEnv
from torch.utils.data import Dataset, DataLoader
from dgllife.model import DGMG
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
import torch
from datetime import timedelta

def cal_time(since):
    now = time.time()
    s = now - since

    if s > 3600:
        h = math.floor(s / 3600)
        m = math.floor((s - h * 3600) / 60)
        s = s - h * 3600 - m * 60
        out = '{}h {}m {:.0f}s'.format(h, m, s)
    else:
        m = math.floor(s / 60)
        s = s - m * 60
        out = '{}m {:.0f}s'.format(m, s)
    return out

def load_smiles_from_file(fname):
    with open(fname) as f:
        smiles = f.read().splitlines()
    return smiles

def standardize_smi_to_mol(smi):
    # Load SMILES to Mol
    mol = Chem.MolFromSmiles(smi)

    # Kekulize
    Chem.Kekulize(mol, clearAromaticFlags=True)

    # neuralization reactions
    patts = (
        # Imidazoles
        ('[n+;H]', 'n'),
        # Amines
        ('[N+;!H0]', 'N'),
        # Carboxylic acids and alcohols
        ('[$([O-]);!$([O-][#7])]', 'O'),
        # Thiols
        ('[S-;X1]', 'S'),
        # Sulfonamides
        ('[$([N-;X2]S(=O)=O)]', 'N'),
        # Enamines
        ('[$([N-;X2][C,N]=C)]', 'N'),
        # Tetrazoles
        ('[n-]', '[n]'),
        # Sulfoxides
        ('[$([S-]=O)]', 'S'),
        # Amides
        ('[$([N-]C=O)]', 'N'),
    )
    reactions = [(Chem.MolFromSmarts(x), Chem.MolFromSmiles(y, False)) for x, y in patts]

    for reactant, product in reactions:
        while mol.HasSubstructMatch(reactant):
            mol = AllChem.ReplaceSubstructs(mol, reactant, product)[0]
    
    return mol


def get_atom_and_bond_types(smiles):

    atom_types = set()
    bond_types = set()
    n_smiles = len(smiles)
    print_every = 10000

    for i, s in enumerate(smiles):

        mol = standardize_smi_to_mol(s)

        for atom in mol.GetAtoms():
            a_symbol = atom.GetSymbol()
            if a_symbol not in atom_types:
                atom_types.add(a_symbol)
        
        for bond in mol.GetBonds():
            b_type = bond.GetBondType()
            if b_type not in bond_types:
                bond_types.add(b_type)
        
        if (i + 1) % print_every == 0:
            print('process {} / {}, ...'.format(i + 1, n_smiles))
    print('process {} / {}, done.'.format(i + 1, n_smiles))
    
    return list(atom_types), list(bond_types)


class MoleculeDataset(object):
    '''
    order: random or canonical
    '''
    def __init__(self, tr_file, val_file, order='random', subset_id=0, n_subsets=1):
        super(MoleculeDataset, self).__init__()

        self.tr_file = tr_file
        self.val_file = val_file
        self.order = order
        self.subset_id = subset_id
        self.n_subsets = n_subsets
        self._setup()

    def collate(self, samples):
        assert len(samples) == 1
        return samples[0]

    def _create_a_subset(self, smiles):
        subset_size = len(smiles) // self.n_subsets
        return Subset(smiles[self.subset_id * subset_size: (self.subset_id + 1) * subset_size],
                      self.order, self.env)

    def _setup(self):
        ### modify by hand ###
        # self.atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
        # self.bond_types = [Chem.rdchem.BondType.SINGLE,
        #                    Chem.rdchem.BondType.DOUBLE,
        #                    Chem.rdchem.BondType.TRIPLE]
        tr_smiles = load_smiles_from_file(self.tr_file)
        val_smiles = load_smiles_from_file(self.val_file)
        all_smiles = tr_smiles + val_smiles

        self.atom_types, self.bond_types = get_atom_and_bond_types(all_smiles)
        print(self.atom_types)
        print(self.bond_types)

        self.env = MoleculeEnv(self.atom_types, self.bond_types)
        self.train_set = self._create_a_subset(tr_smiles)
        self.val_set = self._create_a_subset(val_smiles)


class Subset(Dataset):
    def __init__(self, smiles, order, env):
        super(Subset, self).__init__()
        self.smiles = smiles
        self.order = order
        self.env = env
        self._setup()

    def _setup(self):
        smiles_ = []
        mols = []
        for s in self.smiles:
            m = standardize_smi_to_mol(s)
            smiles_.append(s)
            mols.append(m)
        self.smiles = smiles_
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

class Optimizer(nn.Module):
    def __init__(self, lr, optimizer):
        super(Optimizer, self).__init__()
        self.lr = lr
        self.optimizer = optimizer
        self._reset()

    def _reset(self):
        self.optimizer.zero_grad()

    def backward_and_step(self, loss):
        loss.backward()
        self.optimizer.step()
        self._reset()

    def decay_lr(self, decay_rate=0.99):
        self.lr *= decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

class MultiProcessOptimizer(Optimizer):
    def __init__(self, n_processes, lr, optimizer):
        super(MultiProcessOptimizer, self).__init__(lr=lr, optimizer=optimizer)
        self.n_processes = n_processes
    
    def _sync_gradient(self):
        for param_group in self.optimizer.param_groups:
            for p in param_group['params']:
                if p.requires_grad and p.grad is not None:
                    dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)
                    p.grad.data /= self.n_processes

def synchronize(num_processes):
    if num_processes > 1:
        dist.barrier()

def launch_a_process(rank, args, target, minutes=720):
    dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
        master_ip=args['master_ip'], master_port=args['master_port'])
    dist.init_process_group(backend='gloo',
                            init_method=dist_init_method,
                            # If you have a larger dataset, you will need to increase it.
                            timeout=timedelta(minutes=minutes),
                            world_size=args['num_processes'],
                            rank=rank)
    assert torch.distributed.get_rank() == rank
    target(rank, args)

def main(rank, args):

    # dataset = MoleculeDataset(tr_file, val_file, order='canonical')
    dataset = MoleculeDataset(args['tr_file'], args['val_file'], order='canonical', subset_id=rank, n_subsets=args['num_processes'])

    tr_loader = DataLoader(dataset.train_set, batch_size=1, shuffle=True, collate_fn=dataset.collate)
    val_loader = DataLoader(dataset.val_set, batch_size=1, shuffle=True, collate_fn=dataset.collate)

    print(len(tr_loader))
    print(len(val_loader))

    # initialize model
    model = DGMG(atom_types = dataset.atom_types,
                 bond_types = dataset.bond_types,
                 node_hidden_size = args['node_hidden_size'],
                 num_prop_rounds = args['num_propagation_rounds'],
                 dropout = args['dropout'])
    
    # optimizer
    # optimizer = Optimizer(lr = args['lr'],
    #                       optimizer = Adam(model.parameters(), lr=args['lr']))
    optimizer = MultiProcessOptimizer(args['num_processes'], args['lr'],
                                      Adam(model.parameters(), lr=args['lr']))
    
    # training
    for epoch in range(args['epochs']):
        print('Start training:')
        t = time.time()

        model.train()

        for step, data in enumerate(tr_loader):
            log_prob = model(actions=data, compute_log_prob=True)
            prob = log_prob.detach().exp()

            loss_averaged = - log_prob
            prob_averaged = prob

            optimizer.backward_and_step(loss_averaged)

            if (step + 1) % 100 == 0:
                print('training epoch {}, step {} / {}, loss {:.2f}, time {}'.format(
                    epoch+1, step+1, len(tr_loader), loss_averaged.item(), cal_time(t)))
        
        synchronize(args['num_processes'])








if __name__ == '__main__':

    args = {
        'tr_file': 'ChEMBL_DGMG_train.txt',
        'val_file': 'ChEMBL_DGMG_val.txt',
        'node_hidden_size': 128,
        'num_propagation_rounds': 2,
        'dropout': 0.2,
        'lr': 1e-4,
        'epochs': 10,
        'num_processes': 8,
        'master_ip': '127.0.0.1',
        'master_port': '12345',
    }

    # main(
    #     args = args,
    # )


    mp = torch.multiprocessing.get_context('spawn')
    procs = []
    for rank in range(args['num_processes']):
        procs.append(mp.Process(target=launch_a_process, args=(rank, args, main), daemon=True))
        procs[-1].start()
    for p in procs:
        p.join()

