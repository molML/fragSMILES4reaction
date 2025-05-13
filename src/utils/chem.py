from rdkit import Chem

def MolWithoutIsotopes(mol):
    atoms = [atom for atom in mol.GetAtoms() if atom.GetIsotope()]
    for atom in atoms:
    # restore original isotope values
        atom.SetIsotope(0)
    return mol

def RemoveStereoFromSmiles(s, chars=["/","\\"]):
    for c in chars:
        s=s.replace(c,"")
    return s

def preprocessSmiles(sm):
    mol=Chem.MolFromSmiles(sm)
    if not mol:
        return None
    mol = MolWithoutIsotopes(mol)
    sm = Chem.MolToSmiles(mol)
    sm = RemoveStereoFromSmiles(sm) ## default removes geometric stereochemistry only
    return Chem.CanonSmiles(sm)

def isolate_rings(mol):
    from chemicalgof.reduce import Decompositer
    SINGLEXOCYCLICPATT = Decompositer.SINGLEXOCYCLICPATT # to recognize rings
    bondMatches = mol.GetSubstructMatches( Chem.MolFromSmarts(SINGLEXOCYCLICPATT) )
    bonds=[mol.GetBondBetweenAtoms(*b).GetIdx() for b in bondMatches]
    if not bonds:
        return tuple()
    frags = Chem.FragmentOnBonds(mol, addDummies=False, bondIndices=bonds, )
    fragsMol=Chem.GetMolFrags(frags,asMols=True)
    cycles = [ frag for frag in fragsMol if frag.HasSubstructMatch(Chem.MolFromSmarts('[R]'))]
    cycles_smi = [Chem.MolToSmiles(mol) for mol in cycles ]
    return tuple(cycles_smi)

smarts_side = '[*!R]-[*R]'
def isolate_side_chains(mol):
    bondMatches = mol.GetSubstructMatches( Chem.MolFromSmarts(smarts_side) )
    bonds=[mol.GetBondBetweenAtoms(*b).GetIdx() for b in bondMatches]
    if not bonds:
        return tuple()
    frags = Chem.FragmentOnBonds(mol, addDummies=False, bondIndices=bonds, )
    fragsMol=Chem.GetMolFrags(frags,asMols=True)
    sides = [ frag for frag in fragsMol if not frag.HasSubstructMatch(Chem.MolFromSmarts('[R]'))]
    sides_smi = [Chem.MolToSmiles(mol) for mol in sides ]
    return tuple(sides_smi)