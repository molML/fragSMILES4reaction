params = {
    'temperature': { # XXX this is not employed, actually.
        'type': float,
        'default': 1,
        'help': 'Temperature for sampling softmax'
    },
    'k': {
        'type': float,
        'default': 5,
        'help': 'Store top K predicted tokens'
    },
}