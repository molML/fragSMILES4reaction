from rdkit import Chem
from .error_handling import check_syntax, parse_exception, fragsmiles_substrings, rdkit_substrings, safe_substrings

def _last_chiral_check(smi, canon_smi) -> bool:
    import re
    pattern_stereo = r'@{1,2}'
    canon_stereo_counts = len(re.findall(pattern_stereo, canon_smi))
    gen_stereo_count = len(re.findall(pattern_stereo, smi))

    return canon_stereo_counts == gen_stereo_count

def GenSmiles2Smiles(smi, strict_chirality=False, error_handling=False):
    x = Chem.MolFromSmiles(smi, sanitize=False)
    if not x and not error_handling:
        return None
    elif not x:
        return check_syntax(smi)

    try:
        Chem.SanitizeMol(x)
    except Exception as e:
        if error_handling:
            return parse_exception(str(e), rdkit_substrings)
        return None
    
    canon_smi = Chem.MolToSmiles(x)

    if not strict_chirality:
        return canon_smi

    if _last_chiral_check(smi, canon_smi):
        return canon_smi
    elif not error_handling:
        return None
    else:
        from .error_handling import CHIRAL
        return CHIRAL

def GenSelfies2Smiles(x, strict_chirality=False, error_handling=False):
    import selfies as sf
    try:
        x = sf.decoder(x)
    except Exception as e:
        if error_handling:
            return parse_exception(str(e), rdkit_substrings)
        return None
    
    return GenSmiles2Smiles(x, strict_chirality=strict_chirality, error_handling=error_handling)

# NOTE character "." was replaced temporarly by "&" because of the separator for reactants. It is replaced in this step to decode SAFE to SMILES
def GenSafe2Smiles(x, strict_chirality=False, error_handling=False): # NOTE first, replace & with .
    from safe import decode as safe_decode
    x = x.replace('&','.')
    try:
        x = safe_decode(x)
    except Exception as e:
        if error_handling:
            return parse_exception(str(e), safe_substrings | rdkit_substrings)
        return None
    
    # BUG : SAFE package does not provide error handling ... None is returned instead
    # SAFE code package has been edited for this purpose and stored in this repository.
    if not x:
        # return None
        from .error_handling import DEFAULT
        return DEFAULT

    return GenSmiles2Smiles(x, strict_chirality=strict_chirality, error_handling=error_handling)
    
def GenFragSmiles2Smiles(x:list[str], strict_chirality=False, error_handling=False):
    from chemicalgof import decode
    try:
        x=decode(x, strict_chirality=strict_chirality)
    except Exception as e:
        if error_handling:
            return parse_exception(str(e), fragsmiles_substrings | rdkit_substrings)
        return None
    
    canon_smi = Chem.CanonSmiles(x)

    if not strict_chirality:
        return canon_smi
    
    if _last_chiral_check(x, canon_smi):
        return canon_smi
    elif not error_handling:
        return None
    else:
        from .error_handling import CHIRAL
        return CHIRAL