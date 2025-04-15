import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..')) # NOTE allowing import of src module

def convert_prediction(args):
    # Source
    from src.paths import vocab_path_from_args, pred_path_from_args
    from src.dataloader import  SEP, BOS, EOS
    from src.language_conversion import decoding_fnc_of

    # PyTorch
    import torch

    # useful packages
    import pandas as pd

    DIR_PRED = pred_path_from_args(args)
    if not os.path.exists(DIR_PRED + '_tokens.csv'):
        print('Predicted tokens do not exist. Conversion aborted !')
        return 0

    ## Load Vocabulary
    DIR_VOCAB = vocab_path_from_args(args)
    if os.path.exists( DIR_VOCAB ):
        vocab = torch.load(DIR_VOCAB)
    else:
        print('No vocabulary recognized. Conversion aborted !')
        return 0

    predictions = pd.read_csv(DIR_PRED+'_tokens.csv', header=None)\
                    .map(lambda x: x.split(SEP))\
                    .map(pd.Series, dtype=int)\
                    .map(lambda x: x.tolist())\
                    .map(vocab.translate_iterable)\
                    .map(lambda x: [element for element in x if element!=BOS and element!=EOS ])

    fnc_convert = decoding_fnc_of(args.notation, strict_chirality=True)

    if args.task == 'forward':
        if args.notation != 'fragsmiles': # if notation is not word level, strings are concatenated
            predictions = predictions.map(''.join)

        predictions = predictions.map(fnc_convert)

    elif args.task == 'backward':
        from src.utils.fix_data import adjust_backward_prediction
        word_level = args.notation == 'fragsmiles'
        predictions = predictions.map(lambda x: adjust_backward_prediction(x, fnc_convert, word_level))

    predictions.to_csv(DIR_PRED + '_mols_strict.csv', index=False, header=False)

if __name__ == '__main__':

    from src.configs import get_parser

    parser = get_parser()

    args, unk = parser.parse_known_args()
    if unk:
        print('Unknown arguments passed !')
        print(*unk)

    exit = convert_prediction(args)