from ..configs.constants import SEP
import torch


def flatDF_from_tensor3d(tensor3d, remove=[], as_type=int):
    import pandas as pd
    L,B,K=tensor3d.shape
    df = pd.DataFrame( tensor3d.reshape(L,B*K) ).T
    df = df.where(~df.isin(remove))
    if as_type:
        df = df.apply(lambda x: SEP.join( x.dropna().astype(as_type).astype(str)), axis=1)
    else:
        df = df.apply(lambda x: SEP.join( x.dropna().astype(str)), axis=1)
    df = pd.DataFrame( df.values.reshape((B,K)) )
    return df

def pad3Dtensors(tensors, value, dim=1):
    max_len = max(tensor.size(0) for tensor in tensors)
    return torch.cat([
        torch.nn.functional.pad(tensor, pad=(0, 0, 0, 0, 0, max_len - tensor.size(0)), value=value)
        for tensor in tensors
    ], dim=dim)


## Backward predictions functions ##
def group_sub_reactants(reactants):
    sub_seq=[]
    tmp = []
    for element in reactants:
        if element == '.' or element == '>':
            if tmp:
                sub_seq.append(tmp)
                tmp = []
        else:
            tmp.append(element)
    
    if tmp:
        sub_seq.append(tmp)
    
    return sub_seq

def split_reactants_reagents(seq):

    reagents = []
    reactants = []
    while len(seq) > 0:
        element = seq.pop()
        if element.startswith('A_'):
            reagents.append(element)
        elif element != '>':
            reactants.append(element)
    
    return reactants, reagents

def adjust_backward_prediction(seq:list, fnc_convert, word_level=False):
    reactants, reagents = split_reactants_reagents(seq)

    reactants = group_sub_reactants(reactants)

    reactants_conv = []
    for gen in reactants:
        if not word_level:
            reactant = fnc_convert(''.join(gen))
        else:
            reactant = fnc_convert(gen)

        if reactant:
            reactants_conv.append(reactant)

    smis = reactants_conv + reagents

    return SEP.join(smis)