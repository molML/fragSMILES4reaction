import torch
from torch.utils.data import Dataset as DatasetAbstract
from torch.nn.utils import rnn

import pandas as pd
import numpy as np

from .configs.constants import BOS, EOS, PAD, UNK, SEP, LENGTHS_COSTRAINT

RENAMERS = {
    'forward': {'reactants':'Source', 'product':'Target'},
    'backward': {'reactants':'Target', 'product':'Source'}
}

class DatasetReactions(DatasetAbstract):
    def __init__(self, src_tgt: pd.DataFrame, bos: str=BOS, eos: str=EOS, sep=SEP, task='forward', lengths=None):

        # task leads to setup source and target for model (which one possess special tokens BOS EOS)
        src_tgt = src_tgt.rename(columns=RENAMERS[task])

        # csv of lengths allows to filter out many reactions.
        if lengths is not None:
            masks = [
                lengths[notation] <= lenght
                for notation,lenght in LENGTHS_COSTRAINT.items()
            ]

            mask_union = np.logical_and(*masks)
            src_tgt = src_tgt[mask_union].reset_index(drop=True)

        ## check for special tokens, first
        if bos:
            src_tgt['Target'] = bos + sep + src_tgt['Target']
        if eos:
            src_tgt['Target'] = src_tgt['Target'] + sep + eos

        src_tgt = src_tgt.map(lambda x : x.split(sep))

        src_tgt['length'] = src_tgt.apply( lambda x : max ( len(x['Source']), len(x['Target']) ), axis=1 )

        self.data = src_tgt[['Source','Target']]
        self.lengths = src_tgt['length']
        self._len = len(self.data)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return self.data.values[idx]

def collate_fn_reactions(src_tgt, vocab, pad: str = PAD):
    src_tgt = pd.DataFrame(src_tgt)
    tensors = src_tgt.map(vocab.translate_iterable) ## convert to integers
    tensors = tensors.map(torch.tensor, dtype=torch.int64, ) ## tensorize

    idxPad = vocab[pad]

    return (rnn.pad_sequence(tensors[0].tolist(), batch_first=False, padding_value=idxPad),
            rnn.pad_sequence(tensors[1].tolist(), batch_first=False, padding_value=idxPad))

class Vocab:

    def __init__(self) -> None:
        self.unk = None
        self.i2s={}
        self.s2i={}

    @classmethod
    def build_vocab_from_iterator(cls, iterator, bos: str=BOS, eos: str=EOS, unk: str=UNK, pad: str = PAD, specials=[]):
        from collections import Counter
        obj = cls()

        vocab = []
        if specials:
            vocab.extend(specials)
        vocab.extend( [pad,unk] )
        if bos:
            vocab.append(bos)
        if eos:
            vocab.append(eos)
        
        _counter = Counter(iterator)
        ## remove special tokens if they are already provided in the dictionary vocab
        for key in vocab: _counter.pop(key, None)

        vocab.extend ( sorted(_counter, key=_counter.get, reverse=True) )

        obj.i2s |= {k:v for k,v in enumerate(vocab)}
        obj.s2i |= {k:v for v,k in enumerate(vocab)}
        obj.unk = unk

        return obj

    def __len__(self):
        return len(self.i2s)

    @property
    def default(self):
        return self.s2i[self.unk]

    def __getitem__(self, key: str | int):
        if isinstance(key, str):
            return self.s2i.get(key, self.default)
        elif isinstance(key, int):
            return self.i2s.get(key, self.unk)
        else:
            raise TypeError('translation supports only str and int')
        
    def translate(self, key):
        return self[key]

    def translate_iterable(self, keys):
        return [ self[key] for key in keys ]
    