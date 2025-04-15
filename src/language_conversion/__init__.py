from .error_handling import ERRORS

from .encoding import smi2fragsmiles, smi2safe, smi2selfies, tokenize_smiles
# Function to encode and return splitted tokens by whitespace as a string
def encoding_fnc_of(notation:str)-> 'function':
    notation = notation.lower()
    if notation == 'smiles':
        return tokenize_smiles
    elif notation == 'selfies':
        return smi2selfies
    elif notation == 'safe':
        return smi2safe
    elif notation == 'fragsmiles':
        return smi2fragsmiles
    else:
        raise 'Invalid notation name'
    
from .decoding import GenFragSmiles2Smiles, GenSafe2Smiles, GenSelfies2Smiles, GenSmiles2Smiles
# Function to encode and return splitted tokens by whitespace as a string
def decoding_fnc_of(notation:str, strict_chirality=False, error_handling=False)-> 'function':
    from functools import partial
    notation = notation.lower()

    if notation == 'smiles':
        fnc = GenSmiles2Smiles
    elif notation == 'selfies':
        fnc = GenSelfies2Smiles
    elif notation == 'safe':
        fnc = GenSafe2Smiles
    elif notation == 'fragsmiles':
        fnc = GenFragSmiles2Smiles
    else:
        raise 'Invalid notation name'
    
    return partial(fnc, error_handling=error_handling, strict_chirality=strict_chirality )