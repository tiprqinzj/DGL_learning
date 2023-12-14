import time, math, random
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt


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

def check_symbol(mol):
    
    common_symbols = ['C', 'H', 'N', 'O', 'S', 'F', 'Cl', 'Br']

    all_symbols = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in all_symbols:
            all_symbols.append(atom.GetSymbol())
    
    if 'C' not in all_symbols:
        return False
    
    others = []
    for s in all_symbols:
        if s not in common_symbols:
            others.append(s)
    
    if len(others) > 0:
        return False
    else:
        return True

def check_mw(mol, MIN=200, MAX=800):

    MW = ExactMolWt(mol)

    if MW < MIN or MW > MAX:
        return False
    else:
        return True

def pipe_check_smi(s):

    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return (False, 'Failed in Chem.MolFromSmiles')
    except:
        return (False, 'Failed in Chem.MolFromSmiles')
    
    # check symbol
    if check_symbol(mol) == False:
        return (False, 'Failed because unexpected symbols')
    
    # check MolWt
    if check_mw(mol) == False:
        return (False, 'Failed because unsuitable MW, only MW in range [200, 800] will be remained')
    
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
    
    return (True, mol)


def main(download_file):

    t = time.time()

    # load data
    df_init = pd.read_table(download_file)
    print('Load initial data: {}'.format(df_init.shape[0]))

    # extract smiles
    smiles = df_init.loc[:, 'Smiles'].tolist()
    
    # remove empty data
    smiles = [s for s in smiles if type(s) == str]
    print('After remove empty smiles: {}'.format(len(smiles)))

    # remove mixture
    smiles = [s for s in smiles if '.' not in s]
    print('After remove mixture: {}'.format(len(smiles)))

    # check smi
    unique_smi, unique_inchikey = [], []
    for i, s in enumerate(smiles):
        checked, mol = pipe_check_smi(s)

        if checked:
            smi = Chem.MolToSmiles(mol, isomericSmiles=False)
            inchikey = Chem.MolToInchiKey(mol)[:14]

            if inchikey not in unique_inchikey:
                unique_smi.append(smi)
                unique_inchikey.append(inchikey)
        
        if (i + 1) % 50000 == 0:
            print('process {} / {}, unique {} ({}) ...'.format(i+1, len(smiles), len(unique_smi), cal_time(t)))
    
    print('process {} / {}, unique {} ({}), done.'.format(i+1, len(smiles), len(unique_smi), cal_time(t)))

    # shuffle smiles
    random.seed(42)
    random.shuffle(unique_smi)

    # training samples
    # tr_smiles = unique_smi[:-100000]
    # val_smiles = unique_smi[-100000:]
    # print('After shuffle, training {}, validating {}'.format(len(tr_smiles), len(val_smiles)))

    # write
    # with open('chembl_2023-12-12_train.txt', 'w') as f:
    #     for s in tr_smiles:
    #         f.write(s + '\n')
    
    # with open('chembl_2023-12-12_valid.txt', 'w') as f:
    #     for s in val_smiles:
    #         f.write(s + '\n')
    
    with open('chembl_2023-12-12_unique.txt', 'w') as f:
        for s in unique_smi:
            f.write(s + '\n')

    
    print('Done')


if __name__ == '__main__':

    main('download_allcompds_1920643.tsv')
