params = {
    'batch_size': {
        'type': int,
        'default': 256,
        'help': 'Batch size for training'
    },
    'lr': {
        'type': float,
        'default': 0.001,
        'help': 'Learning rate for training'
    },
    'dropout': {
        'type': float,
        'default': 0.3,
        'help': 'Dropout for each component of architecture'
    },
    'warmup': {
        'type': int,
        'default': 100,
        'help': 'Warmup of learning rate scheduler for training if --cosine is passed'
    },
    'max_iters': {
        'type': int,
        'default': 1000,
        'help': 'Max iterations for stopping learning rate scheduler for training if --cosine is passed'
    },
}