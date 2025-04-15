from .model import TransformerModel, TransformerReactions, TransformerCosine

def model_init_from_args(args):
    from ..configs import model_params, train_params
    from ..configs.constants import LENGTHS_MAX_NOTATION

    keys = list( (model_params | train_params).keys() )

    args_dict = vars(args)
    model_init = {key : args_dict[key] for key in keys if key in args_dict}

    # NOTE Set predefined max length for positional encoder lead to specific notation
    model_init['max_len'] = LENGTHS_MAX_NOTATION[args.notation]

    return model_init