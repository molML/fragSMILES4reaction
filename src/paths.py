import os
import re
from argparse import Namespace


_epochCompiler = re.compile(r'(?<=epoch=)\d+(?=_)')

def epoch_from_checkpoint_string(checkpoint:str) -> int:
    match = _epochCompiler.search(checkpoint)
    return int( match.group() )

def last_checkpoint_from_path(path:str) -> str|None:
    models = [file for _, _, files in os.walk(path) 
            for file in files if file.endswith(".ckpt")]
    
    if not models:
        return None

    return max(models, key=epoch_from_checkpoint_string)

def root_path_from_args(args: Namespace) -> str:
    return os.path.join(args.dir_experiments,
                        'task='+args.task,
                        'notation='+args.notation)

def data_path_from_args(args:dict[str, Namespace], split:str|None = None):
    return os.path.join(args.dir_data, 
                        ('{}_' if not split else f'{split}_') + args.notation)

def model_name_from_args(args):
    from .configs import model_params
    args_dict = vars(args)
    return '-'.join([ f'{key}={args_dict[key]}' for key in model_params.keys() ] )

def train_name_from_args(args):
    from .configs import train_params
    args_dict = vars(args)
    return '-'.join([ f'{key}={args_dict[key]}' for key in train_params.keys() if key in args_dict ])

def setup_name_from_args(args: dict[str, Namespace]) -> str:
    model_name = model_name_from_args(args)
    train_name = train_name_from_args(args)
    return os.path.join(model_name,train_name)

def setup_path_from_args(args: dict[str, Namespace]) -> str:
    setup_name = setup_name_from_args(args)
    root_path = root_path_from_args(args)
    return os.path.join(root_path, setup_name)

def vocab_path_from_args(args: dict[str, Namespace]) -> str:
    return os.path.join(root_path_from_args(args),'vocab.pt')

def pred_path_from_args(args: dict[str, Namespace]) -> str:
    ## retured string misses last piece : `_tokens.csv` or `_molecules.csv`
    return os.path.join(
        setup_path_from_args(args),
        'predictions' + '_top' + str(args.k))

def import_function(file:str, function:str):
    import importlib
    module = importlib.import_module(file)
    return getattr(module, function)

## [ ] add parsing parameters by reading yaml file ?
def kwargs_from_path(path:str):
    relevant_path, ext = os.path.splitext(path)

    compiler_kwargs = re.compile(r'(?P<key>[^-/]+)=(?P<value>[^-/]+)')
    kwargs = compiler_kwargs.findall(relevant_path)
    return dict(kwargs)
