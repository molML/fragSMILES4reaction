params = {
    'model_dim': {
        'type': int,
        'default': 256,
        'help': 'Embedding dimension of vocabulary tokens'
    },
    'num_heads': {
        'type': int,
        'default': 4,
        'help': 'Number of multihead attention per encoding layer'
    },
    'num_layers': {
        'type': int,
        'default': 2,
        'help': 'Number of encoding and/or decoding layers'
    },
}