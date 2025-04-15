from .general_params import params as general_params
from .model_params import params as model_params
from .train_params import params as train_params
from .sample_params import params as sample_params

groups_params = {
    'common': general_params,
    'model': model_params,
    'train': train_params,
    'sample': sample_params
}

import argparse

class ArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, *args,**kwargs):
        args, argv = super(ArgumentParser, self).parse_known_args(*args, **kwargs)

        ## Remove arguments concerning cosine scheduler
        if not args.cosine:
            del args.warmup
            del args.max_iters

        return args, argv
    
def get_parser():

    parser = ArgumentParser(description="Parameters settings")

    for group, group_dict in groups_params.items():
        group_obj = parser.add_argument_group(group)
        for arg, arg_dict in group_dict.items():
            group_obj.add_argument('--'+arg, **arg_dict)

    return parser

def args_from_kwargs(kwargs):
    from itertools import chain
    parser = get_parser()
    kwargs = [ ('--'+k,v) for k,v in kwargs.items() ]
    args,unk = parser.parse_known_args(chain(*kwargs))
    if unk:
        print("unknown:",*unk)

    return args