DEFAULT = 'Misc.'
BONDS = 'Bond Assignment'
KEKULE = 'Kekulization'
BRACKETS = 'Brackets'
RINGS = 'Rings'
CHIRAL = 'Chirality'

ERRORS = [RINGS, KEKULE, BONDS, BRACKETS, CHIRAL, DEFAULT] # order for chart representation

def brackets(smi):
    return smi.count('(') == smi.count(')')

def rings(smi):
    import re
    from collections import Counter

    matches = re.findall(r'(\%[0-9]{2}|[0-9])', smi)
    if not matches:
        return False
    
    counts = Counter(matches)

    return all(count%2==0 for count in counts.values())

def check_syntax(smiles):
    if not brackets(smiles):
        return BRACKETS
    elif not rings(smiles):
        return RINGS
    else:
        return DEFAULT

rdkit_substrings = {
    'explicit valence for atom':BONDS,
    'can\'t kekulize mol':KEKULE, # or Rings
}

fragsmiles_substrings = {
    'bond error' : BONDS, # connector index is not an actual atom connector of a fragment
    'chirality error': CHIRAL, # 'stereocenter assigned to an atom with less than 4 substituents'
    'branching error' : BRACKETS, # 'tag atom index is not explicited for a fragment'
}

safe_substrings = {
    'extra open parentheses' : BRACKETS,
    'ring closure' : RINGS,
    'duplicated ring': RINGS,
    'non-ring':RINGS,
}

def parse_exception(exception, substrings):
    for substring, label in substrings.items():
        if substring.lower() in exception.lower():
            return label
        
    return DEFAULT