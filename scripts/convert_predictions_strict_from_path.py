if __name__ == '__main__':

    from src.configs import args_from_kwargs
    from src.paths import kwargs_from_path
    from convert_predictions_strict import convert_prediction
    import sys
    import os
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, '..')) # NOTE allowing import of src module

    path = sys.argv[1]

    kwargs = kwargs_from_path(path)
    args = args_from_kwargs(kwargs)

    exit = convert_prediction(args)