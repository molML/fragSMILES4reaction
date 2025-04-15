import os
import sys
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..')) # NOTE allowing import of src module

from functools import partial
import pandas as pd

from src.language_conversion import encoding_fnc_of
from src.configs.constants import SEP, NOTATIONS
from src.utils.chem import preprocessSmiles
from src.utils.pool import applyFncPool, apply_on_list

def get_parser():
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Parameters settings")
    parser.add_argument('--dir_rawdata', type=str, default='rawdata_reactions', help='Directory of raw data reactions')
    parser.add_argument('--dir_data', type=str, default='data_reactions', help='Directory of converted data reactions')
    parser.add_argument('--split', type=str, default='train', help='Split of raw data to be converted')
    parser.add_argument('--notation', type=str, default='smiles',choices=NOTATIONS, help='Notation to obtain from raw data reacions')
    parser.add_argument('--ext_output', type=str, default='tar.xz', help='File extension for output converted dataset (e.g. csv or tar.xz)')
    parser.add_argument('--cpus', type=int, default=10, help='Number of CPUs to adopt for multiprocessing')

    return parser

EXTENSION_COMP={
    'tar.xz':'xz',
    # 'csv':None,
}

######
## notation-independent
def getReagentsCol(source):
    sourceSplt = source.str.extract('(?P<reactants>[^>]+)(?P<sep> > ?)(?P<reagents>[^>]*)')
    sourceSplt.fillna('', inplace=True)
    
    return sourceSplt.reagents

def convert(args):

    kwargs_csv = {
        'sep':'\t',
        'usecols':[
            'Source', ## needs only for reagents shared by any notation
            'CanonicalizedReaction', ## necessary to process reactants and product
            ]
                }
    
    # NOTE for an easy github repository: train data takes more than 100MB of memory
    if args.split == 'train':
        format_file = 'tar.xz'
        kwargs_csv['compression'] = 'xz'
    else:
        kwargs_csv['skiprows'] = 2
        format_file = 'csv'

    rawData =  pd.read_csv(os.path.join(args.dir_rawdata,
                f'US_patents_1976-Sep2016_1product_reactions_{args.split}.{format_file}'), **kwargs_csv)

    ## Shared reagents
    reagents = rawData.Source
    rawData.drop(columns='Source', inplace=True)
    reagents = getReagentsCol(reagents)

    splitted = rawData.CanonicalizedReaction.str.extract('(?P<reactants>[^>]+)(?P<sep1>>)(?P<reagents>[^>]*)(?P<sep2>>)(?P<product>[^>]+)')

    ## main reactants elaborated. Keep each smiles as element in list
    splitted['reactants'] = splitted['reactants'].str.split('.')
    splitted['reactants'] = applyFncPool(
                                splitted['reactants'], 
                                fnc=partial(apply_on_list, fnc = preprocessSmiles), ## preprocess each smiles in list and parallelizing each row
                                cpus=args.cpus
                                      )
    
    ## main product elaborated
    splitted['product'] = applyFncPool(splitted['product'], fnc = preprocessSmiles, cpus=args.cpus)

    ## Notation dependent

    notation_fnc = encoding_fnc_of(args.notation)

    reactants = applyFncPool(
                    splitted['reactants'], 
                    fnc=partial(apply_on_list, fnc = notation_fnc),
                    cpus=args.cpus
                            )
    
    reactants = pd.Series(reactants).apply(' . '.join)
    product = applyFncPool(splitted['product'], fnc=notation_fnc, cpus=args.cpus)

    dataset = pd.DataFrame.from_dict(
        {
            'reactants':reactants + SEP + '>' + 
                reagents.apply(lambda x: SEP + x if x else x), ## reaction with no reagents doesnt need SEP arator
            'product':product
        }
        )
    return dataset

if __name__ == "__main__":
    parser = get_parser()
    args, unk = parser.parse_known_args()
    
    dataset = convert(args)

    os.makedirs(args.dir_data, exist_ok=True)
    dataset.to_csv(
                    os.path.join(args.dir_data,f'{args.split}_{args.notation}.{args.ext_output}'),
                    index=False,
                    compression=EXTENSION_COMP.get(args.ext_output,None), 
                    sep='\t'
    )

    path_lengths_file = os.path.join(args.dir_data,f'{args.split}_lengths.csv')
    if os.path.exists( path_lengths_file ):
        lengths = pd.read_csv(path_lengths_file)
    else:
        lengths = pd.DataFrame(dtype=int)

    # computing lengths
    tokenized = dataset.map(lambda x: x.split(SEP))

    # NOTE +2 because they needs bos and eos tokens (actually only 1 of them: it depends from what the Target is !)
    lengths[args.notation] = tokenized.apply( lambda x : max ( len(x['reactants']) + 2, len(x['product']) +2 ), axis=1 ) 

    lengths.to_csv(path_lengths_file, index=False)
