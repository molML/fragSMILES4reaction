import os
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..')) # NOTE allowing import of src module

import pandas as pd

def get_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Parameters settings")
    parser.add_argument('--dir_data', type=str, default='data_reactions', help='Directory of converted data reactions')
    parser.add_argument('--split', type=str, default='train', help='Split of raw data to be converted')
    parser.add_argument('--ext_output', type=str, default='csv', help='File extension for output converted dataset (e.g. csv or tar.xz)')
    parser.add_argument('--cpus', type=int, default=10, help='Number of CPUs to adopt for multiprocessing')

    return parser

EXTENSION_COMP={
    'tar.xz':'xz',
    # 'csv':None,
}

def fragment(args):
    from src.configs.constants import SEP
    from src.utils.chem import isolate_rings, isolate_side_chains

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    data_smi =  pd.read_csv(os.path.join(args.dir_data,f'{args.split}_smiles.tar.xz'), compression='xz', sep='\t').map(lambda x: x.replace(SEP,''))

    substructures = pd.DataFrame()
    mols = data_smi['product'].apply(Chem.MolFromSmiles)

    substructures['cyclic'] = mols.apply(isolate_rings)
    substructures['acyclic'] = mols.apply(isolate_side_chains)
    substructures['scaffold'] = data_smi['product'].apply(lambda x: MurckoScaffold.MurckoScaffoldSmiles(smiles=x, includeChirality=True) )

    substructures = substructures.map(lambda x: "" if not x else(' '.join(x) if type(x) is tuple else x))


if __name__ == "__main__":
    parser = get_parser()
    args, unk = parser.parse_known_args()
    
    substructures = fragment(args)

    path_file = os.path.join(args.dir_data,f'{args.split}_fragments.{args.ext_output}')
    substructures.to_csv(
        path_file, 
        index=False,
        compression=EXTENSION_COMP.get(args.ext_output,None),
    )