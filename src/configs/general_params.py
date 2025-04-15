from .constants import NOTATIONS as _notations

params = {
    'dir_experiments': {
        'type': str,
        'default': './experiments_reactions',
        'help': 'Directory of general experiments'
    },
    'dir_data': {
        'type': str,
        'default': './data_reactions',
        'help': 'Directory of general dataset'
    },
    'task': {
        'type': str,
        'default': 'forward',
        'choices': ('forward', 'backward'),
        'help': 'Task affecting model Source and Target'
    },
    'notation': {
        'type': str,
        'default': 'fragsmiles',
        'choices': _notations,
        'help': 'Notation employed for model training'
    },
    'cosine': {
        'action': 'store_true',
        'default': False,
        'help': 'Adopt cosine scheduler for learning rate if passed.'
    },
}