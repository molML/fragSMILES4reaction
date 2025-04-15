import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

from .paths import kwargs_from_path, vocab_path_from_args, root_path_from_args, setup_path_from_args, data_path_from_args, pred_path_from_args
from .dataloader import DatasetReactions, SEP

from .configs import args_from_kwargs
from .model import model_init_from_args

import torch

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

fpgen = rdFingerprintGenerator.GetMorganGenerator()

def _compute_similarity(tgt, smiles_ref, fp_ref):
    if pd.isna(tgt):
        return tgt
    
    elif tgt == smiles_ref:
        # return -1.0 # NOTE we prefer return +1.0 for exact match
        return 1.0
    
    fp_tgt = fpgen.GetCountFingerprint(Chem.MolFromSmiles(tgt))
    return DataStructs.TanimotoSimilarity(fp_ref, fp_tgt) 


def _overlap_mol_similarity(true_predSmiles):
    true_smiles = true_predSmiles.Target

    true_mol = Chem.MolFromSmiles(true_smiles)
    true_fp = fpgen.GetCountFingerprint(true_mol)

    pred_smiles = true_predSmiles.iloc[1:]
    similarities = pred_smiles.apply(_compute_similarity, smiles_ref = true_smiles, fp_ref=true_fp)
    return similarities


class Evaluator:
    test_lengths = pd.DataFrame()
    vocabs = {}

    @classmethod
    def from_path(cls, path):

        kwargs = kwargs_from_path(path)
        return cls.from_dict_args(kwargs)
    
    @classmethod
    def from_dict_args(cls, kwargs):
        
        args = args_from_kwargs(kwargs)

        if args.task == 'forward':
            return ForwardEvaluator(args)
        elif args.task == 'backward':
            return BackwardEvaluator(args)

    # TODO prevent to initialize this Base class.
    def __init__(self, args):
        self.args = args
        self.DIR_VOCAB = vocab_path_from_args(args)
        self.DIR_ROOT = root_path_from_args(args)
        self.DIR_SETUP = setup_path_from_args (args)
        self.DIR_DATA = data_path_from_args(args, split='test')

        self.logs = None
        self.vocab = None

        # [x] to delete ? No : we need true sequence of token (atom or fragments)
        self.true = None
        self.pred_tokens=None

        if self.test_lengths.empty:
            test_lengths = pd.read_csv(
                os.path.join(args.dir_data, 'test_lengths.csv')
                )
            for col in test_lengths.columns:
                self.test_lengths[col] = test_lengths[col]
            del test_lengths

        if self.true_smiles.empty:
            true_smiles = self._load_true_smiles()

            for col in true_smiles.columns:
                self.true_smiles[col] = true_smiles[col]

            del true_smiles


        self.vocab = self.vocabs.get(args.notation)
        if self.vocab is None:
            self.vocab = torch.load(self.DIR_VOCAB)
            self.vocabs[self.args.notation] = self.vocab

    def has_params(self, params:dict):
        self_params = self.args.__dict__
        return self_params.items() >= params.items()

    def load_logs(self):
        path = os.path.join(self.DIR_SETUP,'logs.csv')
        if not os.path.exists(path):
            return False
        
        self.logs=pd.read_csv(path)
        return True

    def load_perplexity(self):
        logits_cum = self.logits_cum = pd.read_csv(os.path.join(self.DIR_SETUP,'predictions_top5_logits_cum.csv'), header=None)
        lengths_pred = self.pred_tokens.map(len)
        logits_norm = logits_cum / lengths_pred
        self.perplexity = (-logits_norm).map(lambda x: np.power(2,x))
        return True

    def load_logits(self):
        df = pd.read_csv(os.path.join(self.DIR_SETUP,'predictions_top5_logits.csv'), header=None)
        df = df.map(lambda x: x.split(SEP))\
                .map( np.array, dtype=float )
        
        self.logits = df
        return True

    def plot_logs(self, ax):
        if self.logs is None:
            return
        
        ax.plot(self.logs[['train_loss','val_loss']])

        textstr='\n'.join(
            [ f'{k}:{v}' for k,v in model_init_from_args(self.args).items()]
                )

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', edgecolor='white', facecolor='none', alpha=1)

        # Place a text box in the upper left in axes coordinates
        ax.text(0.75, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props, color='lightblue')
        pass

    def as_dataframe_cell(self):
        indexes = model_init_from_args(self.args)
        indexes['notation']=self.args.notation
        max_len = indexes.pop('max_len')

        df_cell = pd.DataFrame.from_dict(
            {k:[v] for k,v in indexes.items()} | {'evaluator': [self] }
            )
        
        # df_cell.set_index(list(indexes.keys()), inplace=True)
        
        return df_cell


class BackwardEvaluator(Evaluator):
    ## preloaded objects are stored in dictionaries where
    ## (notation,max_len):object
    vocabs = {} 
    data = {}

    ## preloaded true smiles are stored
    true_smiles = pd.DataFrame()

    def _load_true_smiles(self):
        args = self.args
        true = DatasetReactions(
            pd.read_csv('data_reactions/test_smiles' + '.tar.xz', sep = '\t', compression='xz'),
            # bos=None,
            # eos=None,
            task = args.task,
            lengths=self.test_lengths
            )

        true = true.data.Target
        true = true.apply(lambda x: ''.join(x[1:-1]) )\
                    .str.split('>|\\.', expand=False, regex=True)
        true = true.to_frame()

        true['chiral'] = true.Target.apply(lambda reactants: any('@' in reactant for reactant in reactants))

        return true
    
    def load_true(self):
        args = self.args

        ## Loading DATA test
        self.true = self.data.get(args.notation)
        if self.true is None:
            
            true = DatasetReactions(
            pd.read_csv(self.DIR_DATA + '.tar.xz', sep = '\t', compression='xz'),
            task = args.task,
            lengths=self.test_lengths
            )
            
            true = true.data.Target
            true = true.apply(self.vocab.translate_iterable)\
                    .apply(np.array)
            
            self.true = true
            self.data[args.notation] = self.true

        return True
    
    def load_pred_tokens(self):
        if self.vocab is None or self.true is None:
            self.load_true()

        pred_pth = pred_path_from_args(self.args)
        if not os.path.exists(pred_pth+'_tokens.csv'):
            return False

        df = pd.read_csv(pred_pth+'_tokens.csv', header=None)

        df = df.map(lambda x: x.split(SEP))\
                .map( np.array, dtype=int )
        
        self.pred_tokens = df

        return True
    
    def load_mols(self):
        pred_pth = pred_path_from_args(self.args)
        if not os.path.exists(pred_pth+'_mols.csv'):
            return False

        df = pd.read_csv(pred_pth+'_mols.csv', header=None)\
            .fillna('')\
            .map(lambda x: x.split(SEP))

        self.pred_mols = df

        return True
    
    def load_mols_strict(self):
        pred_pth = pred_path_from_args(self.args)
        if not os.path.exists(pred_pth+'_mols_strict.csv'):
            return False

        df = pd.read_csv(pred_pth+'_mols_strict.csv', header=None)\
            .fillna('')\
            .map(lambda x: x.split(SEP))

        self.pred_mols_strict = df

        return True

    @property
    def valids(self):
        pred_count = self.pred_mols.map(len).values
        true_count = self.true_smiles[['Target']].map(len).values

        score = pred_count/true_count

        # score[score > 1] = 1 # NOTE For not binary valid metric
        score[score < 1] = 0
        
        return score.astype(bool)
    
    @property
    def valids_strict(self):
        pred_count = self.pred_mols_strict.map(len).values
        true_count = self.true_smiles[['Target']].map(len).values

        score = pred_count/true_count

        # score[score > 1] = 1 # NOTE For not binary valid metric
        score[score < 1] = 0
        
        return score.astype(bool)

    
    def validity(self, topk=1, strict_chiral=False):
        if not strict_chiral:
            # return self.valids[:,:topk+1].max(axis=1).sum() # NOTE This was for not binary valid metric (float values)
            return self.valids[:,:topk+1].any(axis=1).sum()
        else:
            return self.valids_strict[:,:topk+1].any(axis=1).sum()

    @property
    def chiral_valids(self):
        return self.valids & self.true_smiles[['chiral']].values
    
    @property
    def chiral_valids_strict(self):
        return self.valids_strict & self.true_smiles[['chiral']].values
    
    def chiral_validity(self, topk=1, strict_chiral=False):
        if not strict_chiral:
            return (self.valids[:,:topk+1].any(axis=1) & self.true_smiles['chiral'].values).sum()
        else:
            return (self.valids_strict[:,:topk+1].any(axis=1) & self.true_smiles['chiral'].values).sum()
    
    @property # TODO delete this method ?
    def accurates_old(self):
        from collections import Counter
        def compare_topk(column):
            return [ 
                len(list((Counter(a) & Counter(b)).elements()))
                for a,b in zip(column, self.true_smiles.Target)
             ]

        self.accurates = self.pred_mols.apply(compare_topk).values / self.true_smiles.map(len).values

    @property
    def accurates(self):
        from collections import Counter
        def compare_topk(column):
            return [ 
                len(list((Counter(a) & Counter(b)).elements())) == len(a)
                for a,b in zip(column, self.true_smiles.Target)
             ]

        return self.pred_mols.apply(compare_topk).values
    
    @property
    def accurates_strict(self):
        from collections import Counter
        def compare_topk(column):
            return [ 
                len(list((Counter(a) & Counter(b)).elements())) == len(a)
                for a,b in zip(column, self.true_smiles.Target)
             ]

        return self.pred_mols_strict.apply(compare_topk).values

    # FIXME metrics could be merged into the Base Class Evaluator since the calc is shared ( NOTE for not binary metric in Backward evaluator !)
    def accuracy(self, topk=1, strict_chiral=False):
        # return self.accurates[:,:topk+1].max(axis=1).sum()
        if not strict_chiral:
            return self.accurates[:,:topk+1].any(axis=1).sum()
        else:
            return self.accurates_strict[:,:topk+1].any(axis=1).sum()
    
    @property
    def chiral_accurates(self):
        return self.accurates & self.true_smiles[['chiral']].values
    
    @property
    def chiral_accurates_strict(self):
        return self.accurates_strict & self.true_smiles[['chiral']].values
    
    def chiral_accuracy(self, topk=1, strict_chiral=False):
        if not strict_chiral:
            return (self.accurates[:,:topk+1].any(axis=1) & self.true_smiles['chiral'].values).sum()
        else:
            return (self.accurates_strict[:,:topk+1].any(axis=1) & self.true_smiles['chiral'].values).sum()


class ForwardEvaluator(Evaluator):
    ## preloaded objects are stored in dictionaries where
    ## notation:object
    vocabs = {} 
    data = {}

    ## preloaded true smiles are stored
    true_smiles = pd.DataFrame()

    def _load_true_smiles(self):
        args = self.args
        true = DatasetReactions(
            pd.read_csv('data_reactions/test_smiles' + '.tar.xz', sep = '\t', compression='xz'),
            # bos=None,
            # eos=None,
            task = args.task,
            lengths=self.test_lengths
            )

        true = true.data.Target
        true = true.apply(lambda x: ''.join(x[1:-1]) )
        true = true.to_frame()
        true['chiral'] = true.Target.str.contains('@')
        return true

    def load_true(self):
        args = self.args

        ## Loading DATA test
        self.true = self.data.get(args.notation)
        if self.true is None:
            
            true = DatasetReactions(
            pd.read_csv(self.DIR_DATA + '.tar.xz', sep = '\t', compression='xz'),
            task = args.task,
            lengths=self.test_lengths
            )
            
            true = true.data.Target
            true = true.apply(self.vocab.translate_iterable)\
                    .apply(np.array)
            
            self.true = true
            self.data[args.notation] = self.true

        return True

    def load_pred_tokens(self):
        if self.vocab is None or self.true is None:
            self.load_true()

        pred_pth = pred_path_from_args(self.args)
        if not os.path.exists(pred_pth+'_tokens.csv'):
            return False

        df = pd.read_csv(pred_pth+'_tokens.csv', header=None)

        df = df.map(lambda x: x.split(SEP))\
                .map( np.array, dtype=int )
        
        self.pred_tokens = df

        return True

    def load_mols(self):
        if self.vocab is None or self.true is None:
            self.load_true()

        pred_pth = pred_path_from_args(self.args)
        if not os.path.exists(pred_pth+'_mols.csv'):
            return False

        df = pd.read_csv(pred_pth+'_mols.csv', header=None)

        self.pred_mols = df

        return True
    
    def load_mols_strict(self):
        if self.vocab is None or self.true is None:
            self.load_true()

        pred_pth = pred_path_from_args(self.args)
        if not os.path.exists(pred_pth+'_mols_strict.csv'):
            return False

        df = pd.read_csv(pred_pth+'_mols_strict.csv', header=None)

        self.pred_mols_strict = df

        return True

    def compute_similarity(self, strict_chiral=False):
        
        pred_mols = self.pred_mols if not strict_chiral else self.pred_mols_strict

        similarities = pd.concat([self.true_smiles[['Target']], pred_mols], axis=1).apply(_overlap_mol_similarity, axis=1)
        return similarities
    
    @property
    def valids(self):
        return self.pred_mols.notnull().values # boolean mask
    
    @property
    def valids_strict(self):
        return self.pred_mols_strict.notnull().values # boolean mask
    
    @property
    def accurates(self):
        return self.pred_mols.values == self.true_smiles[['Target']].values
    
    @property
    def accurates_strict(self):
        return self.pred_mols_strict.values == self.true_smiles[['Target']].values
    
    def validity(self, topk=1, strict_chiral=False):
        if not strict_chiral:
            return self.valids[:,:topk+1].any(axis=1).sum()
        else:
            return self.valids_strict[:,:topk+1].any(axis=1).sum()
    
    @property
    def chiral_valids(self):
        return self.valids & self.true_smiles[['chiral']].values
    
    @property
    def chiral_valids_strict(self):
        return self.valids_strict & self.true_smiles[['chiral']].values
    
    def chiral_validity(self, topk=1, strict_chiral=False):
        if not strict_chiral:
            return (self.valids[:,:topk+1].any(axis=1) & self.true_smiles['chiral'].values).sum()
        else:
            return (self.valids_strict[:,:topk+1].any(axis=1) & self.true_smiles['chiral'].values).sum()
    
    def accuracy(self, topk=1, strict_chiral=False):
        if not strict_chiral:
            return self.accurates[:,:topk+1].any(axis=1).sum()
        else:
            return self.accurates_strict[:,:topk+1].any(axis=1).sum()
    
    @property
    def chiral_accurates(self):
        return self.accurates & self.true_smiles[['chiral']].values
    
    @property
    def chiral_accurates_strict(self):
        return self.accurates_strict & self.true_smiles[['chiral']].values
    
    def chiral_accuracy(self, topk=1, strict_chiral=False):
        if not strict_chiral:
            return (self.accurates[:,:topk+1].any(axis=1) & self.true_smiles['chiral'].values).sum()
        else:
            return (self.accurates_strict[:,:topk+1].any(axis=1) & self.true_smiles['chiral'].values).sum()