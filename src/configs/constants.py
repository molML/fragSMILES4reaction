BOS='<bos>'
EOS='<eos>'
PAD='<pad>'
UNK='<unk>'

SEP=' '

NOTATIONS = ['smiles','selfies','safe','fragsmiles',] # This defines order for visualization in tables and figures

NOTATIONS_NAME = {
    'smiles':'SMILES',
    'fragsmiles':'fragSMILES',
    'safe':'SAFE',
    'selfies':'SELFIES',
}

NOTATIONS_COLOR = {
    'smiles':'#ff8c8c',
    'fragsmiles':'#c1e5f5',
    'safe':'#d9f2d0',
    'selfies':'#E8B86D',
}

LENGTHS_COSTRAINT = {
    'smiles' : 200,
    'fragsmiles' : 150,
}

LENGTHS_MAX_NOTATION = {
    'safe' : 389,
    'selfies' : 208,
} | LENGTHS_COSTRAINT