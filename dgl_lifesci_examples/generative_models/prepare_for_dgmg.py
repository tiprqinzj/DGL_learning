import pickle
from rdkit import Chem
from rdkit.Chem import AllChem

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

    print('Get atom and bond types:')
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
            print('  {} / {}, ...'.format(i + 1, n_smiles))
    print('  {} / {}, done.'.format(i + 1, n_smiles))
    
    return list(atom_types), list(bond_types)


def main(tr_file, val_file, out_types):

    tr_smiles = load_smiles_from_file(tr_file)
    val_smiles = load_smiles_from_file(val_file)
    all_smiles = tr_smiles + val_smiles

    atom_types, bond_types = get_atom_and_bond_types(all_smiles)
    print('Atom types: {}'.format(atom_types))
    print('Bond types: {}'.format(bond_types))

    # save atom and bond types dict to file
    with open(out_types, 'wb') as f:
        pickle.dump({'atom_types': atom_types, 'bond_types': bond_types}, f)
    
if __name__ == '__main__':

    main(
        tr_file = 'ChEMBL_DGMG_train.txt',
        val_file = 'ChEMBL_DGMG_val.txt',
        out_types = 'ChEMBL_DGMG_atom_bond_types.pkl'
    )

