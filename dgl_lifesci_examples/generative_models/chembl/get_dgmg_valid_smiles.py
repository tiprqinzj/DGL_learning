import random, time, math
from rdkit import Chem
from dgllife.model.model_zoo.dgmg import MoleculeEnv

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


def eval_decisions(env, decisions):
    env.reset(rdkit_mol=True)
    t = 0

    def whether_to_add_atom(t):
        assert decisions[t][0] == 0
        atom_type = decisions[t][1]
        t += 1
        return t, atom_type

    def whether_to_add_bond(t):
        assert decisions[t][0] == 1
        bond_type = decisions[t][1]
        t += 1
        return t, bond_type

    def decide_atom2(t):
        assert decisions[t][0] == 2
        dst = decisions[t][1]
        t += 1
        return t, dst

    t, atom_type = whether_to_add_atom(t)
    while atom_type != len(env.atom_types):
        env.add_atom(atom_type)
        t, bond_type = whether_to_add_bond(t)
        while bond_type != len(env.bond_types):
            t, dst = decide_atom2(t)
            env.add_bond((env.num_atoms() - 1), dst, bond_type)
            t, bond_type = whether_to_add_bond(t)
        t, atom_type = whether_to_add_atom(t)
    assert t == len(decisions)

    return env.get_current_smiles()


def get_DGMG_smiles(env, mol):
    canonical_decisions = env.get_decision_sequence(mol, list(range(mol.GetNumAtoms())))
    canonical_smile = eval_decisions(env, canonical_decisions)
    
    order = list(range(mol.GetNumAtoms()))
    random.shuffle(order)
    random_decisions = env.get_decision_sequence(mol, order)
    random_smile = eval_decisions(env, random_decisions)
    
    return canonical_smile, random_smile


def main(in_file, out_file, invalid_file):

    # initialize atom and bond types
    atom_types = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br']
    bond_types = [Chem.rdchem.BondType.SINGLE,
                  Chem.rdchem.BondType.DOUBLE,
                  Chem.rdchem.BondType.TRIPLE]

    # initialize env
    env = MoleculeEnv(atom_types, bond_types)

    # initialize log file
    with open(invalid_file, 'w') as f:
        f.write('in_smi\tcanonical_smi\trandom_smi\n')

    # load smiles
    with open(in_file) as f:
        smiles = f.read().splitlines()

    valid_smiles = []

    # execute
    t = time.time()
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        standard_smi = Chem.MolToSmiles(mol)
        
        canonical_smi, random_smi = get_DGMG_smiles(env, mol)
        
        if (canonical_smi != random_smi) or (standard_smi != canonical_smi):
            with open(invalid_file, 'a') as f:
                f.write('{}\t{}\t{}\n'.format(standard_smi, canonical_smi, random_smi))
        else:
            valid_smiles.append(smi)
        
        if (i + 1) % 50000 == 0:
            print('process {} / {}, valid {} (time {}) ...'.format(i+1, len(smiles), len(valid_smiles), cal_time(t)))
            
            with open(out_file + '.temp', 'w') as f:
                for s in valid_smiles:
                    f.write(s + '\n')

    print('process {} / {}, valid {} (time {}).'.format(i+1, len(smiles), len(valid_smiles), cal_time(t)))

    # write
    with open(out_file, 'w') as f:
        for s in valid_smiles:
            f.write(s + '\n')
    
    print('Done.')

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('-i', '--in_file', type=str, metavar='', help='input SMILES file')
    parser.add_argument('-o', '--out_file', type=str, metavar='', help='output SMILES file')
    parser.add_argument('-e', '--invalid_file', type=str, metavar='', help='output invalid SMILES file')

    # unpacked args
    args = parser.parse_args().__dict__

    # execute
    main(
        in_file = args['in_file'],
        out_file = args['out_file'],
        invalid_file = args['invalid_file']
    )
