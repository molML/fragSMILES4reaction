import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..')) # NOTE allowing import of src module

def predict(args):
    # Source
    from src.paths import root_path_from_args, data_path_from_args, vocab_path_from_args, setup_path_from_args, last_checkpoint_from_path, import_function, pred_path_from_args
    from src.utils.fix_data import pad3Dtensors, flatDF_from_tensor3d
    TransformerModel = import_function('src.model','Transformer' + ('Cosine' if args.cosine else '') + 'Reactions')
    from src.dataloader import  collate_fn_reactions, DatasetReactions, PAD

    # PyTorch
    import torch
    from torch.utils.data import DataLoader

    # PyTorch Lightning
    import lightning as L

    # useful packages
    import pandas as pd
    from functools import partial

    DIR_ROOT = root_path_from_args(args)

    ## Looking for model checkpoint
    DIR_SETUP=setup_path_from_args(args)

    ckpt_name = last_checkpoint_from_path(DIR_SETUP)
    if not ckpt_name:
        print('No checkpoint recognized. Prediction aborted !')
        return 0
    else:
        ckpt_path=os.path.join(DIR_SETUP,ckpt_name)

    DIR_PRED = pred_path_from_args(args)
    if os.path.exists(DIR_PRED + '_tokens.csv') or os.path.exists(DIR_PRED + '_mols.csv'):
        print('Any prediction file already exists. Prediction aborted !')
        return 0

    ## Loading DATA
    DIR_DATA = data_path_from_args(args, split='test')

    test = DatasetReactions(
         pd.read_csv(DIR_DATA + '.tar.xz',
                     compression='xz',
                     sep = '\t',
                    #  nrows=1024, # XXX for dev try
                     ),
                    task = args.task,
                    lengths=pd.read_csv(os.path.join(args.dir_data, 'test_lengths.csv'))
    )

    ## Load Vocabulary
    DIR_VOCAB = vocab_path_from_args(args)
    if os.path.exists( DIR_VOCAB ):
        vocab = torch.load(DIR_VOCAB)
    else:
        print('No vocabulary recognized. Prediction aborted !')
        return 0

    ## Initialize Dataloaders
    collate_fn_vocab=partial(collate_fn_reactions, vocab=vocab)

    testDataloader = DataLoader(test, collate_fn=collate_fn_vocab, 
                                batch_size=args.batch_size,
                                # batch_size=64, # XXX for dev try
                                shuffle=False,
                            )

    model = TransformerModel.load_from_checkpoint(ckpt_path)

    ## [x] do we need temperature ? Not for now
    model.temperature = args.temperature
    model.k = args.k

    trainer = L.Trainer(
        # fast_dev_run=True,
        accelerator="auto",
        strategy="auto",
        num_nodes=1,
        default_root_dir=DIR_ROOT,
        # accelerator="gpu",
        devices=1, # it should be 'auto'
        # gradient_clip_val=5, # NOTE set to default value
        num_sanity_val_steps=0,
        log_every_n_steps=-1, ## to avoid warning in terminal. We want just to track every epoch
        # enable_progress_bar = False, # XXX it disables bar progress... useful for HPC
        enable_checkpointing=False,
        logger=False,
        deterministic=True,
    )

    predict_output = trainer.predict(model, testDataloader, return_predictions=True)
    logits_cum = tuple(dict_.pop('logits_cum') for dict_ in predict_output)
    logits = tuple(dict_.pop('logits') for dict_ in predict_output)
    predictions = tuple(dict_.pop('predictions') for dict_ in predict_output)

    logits_cum = torch.concat(logits_cum, dim=0)
    pd.DataFrame(logits_cum).to_csv(DIR_PRED+'_logits_cum.csv', index=False, header=False)

    logits = pad3Dtensors(logits, value=0.0, dim=1)
    predictions = pad3Dtensors(predictions, value=vocab[PAD], dim=1)

    logits = flatDF_from_tensor3d(logits, remove=[0.0], as_type=False)
    logits.to_csv(DIR_PRED+'_logits.csv', index=False, header=False)

    predictions = flatDF_from_tensor3d(predictions, remove=[vocab[PAD]])
    predictions.to_csv(DIR_PRED+'_tokens.csv', index=False, header=False)

    return 1

if __name__ == '__main__':

    from src.configs import get_parser

    parser = get_parser()

    args, unk = parser.parse_known_args()
    if unk:
        print('Unknown arguments passed !')
        print(*unk)

    exit = predict(args)