from ..dataloader import SEP
import re

# NOTE different fragmentation rules are not provdide for this work. Default rule is employed.
def smi2fragsmiles(sm, sep=SEP, default=None):
    from chemicalgof import Reduce2GoF, GoF2fragSMILES, split
    import networkx as nx

    diG=Reduce2GoF(smiles=sm, capitalize_legacy=True)
    # NOTE bug from rdkit if SMARTS does not recognize exocyclic single bonds
    if list(nx.simple_cycles(diG.to_undirected())):
        return default
    # NOTE augmentation is not provided for this work
    out = GoF2fragSMILES(diG, canonize=True, random=False)
    if sep is not None:
        out= sep.join( split(out) )
    
    return out

# NOTE regex has been edited to handle replaced token by SAFE: '.' -> '&'
regex_smiles= r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9]|&)"

tokenizer_smiles=re.compile(regex_smiles)
def tokenize_smiles(sm, sep = SEP):
    tokens = tokenizer_smiles.findall(sm)
    return sep.join(tokens)

def smi2selfies(sm, sep = SEP):
    import selfies as sf
    selfies = sf.encoder(sm, strict=False)
    if sep is None:
        return selfies
    tokens = sf.split_selfies(selfies)
    return sep.join(tokens)

# NOTE SAFE representation includes '.' to split fragments. This character is in conflict with notation for reactions, so it has been replaced by '&'
def smi2safe(sm, sep=SEP):
    from safe import encode as safe_encode
    try:
        safe = safe_encode(sm)
        safe = safe.replace('.','&')
    except: # NOTE Exception is for molecule composed by only 1 fragment
        # print(sm,'1-frag')
        safe=sm
    if sep is None:
        return safe
    
    tokens = tokenizer_smiles.findall(safe)
    return SEP.join(tokens)