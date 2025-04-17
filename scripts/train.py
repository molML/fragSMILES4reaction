import sys
import os
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..')) # NOTE allowing import of src module

def train(args):
    
    # Source
    from src.dataloader import Vocab, collate_fn_reactions, DatasetReactions
    from src.model import model_init_from_args
    from src.paths import last_checkpoint_from_path, root_path_from_args, setup_name_from_args, import_function, vocab_path_from_args, data_path_from_args, setup_path_from_args, epoch_from_checkpoint_string

    TransformerModel = import_function('src.model','Transformer' + ('Cosine' if args.cosine else '') + 'Reactions')

    # PyTorch
    import torch
    from torch.utils.data import DataLoader

    # PyTorch Lightning
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
    from lightning.pytorch.loggers import CSVLogger
    # from lightning.pytorch.loggers import WandbLogger

    # useful packages
    import pandas as pd
    from functools import partial
    from itertools import chain

    DIR_ROOT = root_path_from_args(args)

    DIR_SETUP=setup_path_from_args(args)
    os.makedirs(DIR_SETUP, exist_ok=True)

    ## Checking for existing checkpoint
    ckpt_name = last_checkpoint_from_path(DIR_SETUP)
    if not ckpt_name:
        ckpt_path=None
    elif 'early_stopping' in ckpt_name:
        print('Training didn\'t resume. Early stopped model recognized !')
        return 0
    ## Cool trick here ;)
    elif last_epoch:=epoch_from_checkpoint_string(ckpt_name) == max_epochs -1:
        print('Max epochs already reach !')
        return 0
    elif last_epoch > max_epochs -1:
        print('Max epochs were over passed !')
        return 0
    else:
        ckpt_path=os.path.join(DIR_SETUP,ckpt_name)

    ## Loading DATA
    DIR_DATA = data_path_from_args(args)

    train = DatasetReactions(
         pd.read_csv(DIR_DATA.format('train') + '.tar.xz', compression='xz', sep = '\t'),
         task = args.task,
         lengths=pd.read_csv(os.path.join(args.dir_data, 'train_lengths.csv'))
         )

    valid = DatasetReactions(
         pd.read_csv(DIR_DATA.format('valid') + '.tar.xz', compression='xz', sep = '\t'),
         task = args.task,
         lengths=pd.read_csv(os.path.join(args.dir_data, 'valid_lengths.csv'))
         )

    ## Build/Load Vocabulary
    DIR_VOCAB = vocab_path_from_args(args)
    if os.path.exists( DIR_VOCAB ):
        vocab = torch.load(DIR_VOCAB)
    else:
        vocab = Vocab.build_vocab_from_iterator(chain(*train.data.values.flatten()))
        torch.save(vocab, DIR_VOCAB)

    ## Initialize Dataloaders
    collate_fn_vocab=partial(collate_fn_reactions, vocab=vocab)

    trainDataloader = DataLoader(train, collate_fn=collate_fn_vocab, 
                            batch_size=args.batch_size,
                            )

    validDataloader = DataLoader(valid, collate_fn=collate_fn_vocab,
                            batch_size=args.batch_size,
                            )

    ## Initialize model
    model = TransformerModel(len(vocab),
                            **model_init_from_args(args),
                            vocab = vocab
                            )

    ## Setting up lightning environment
    # [x] Report to community the replacing of previous logs 
    # ~/envPython/lib/python3.11/site-packages/lightning/fabric/loggers/csv_logs.py -> self._check_log_dir_exists()
    # is a method that remove previous metric file. We don't want this if we are reloading a checkpoint ! Report it :)
    logger = CSVLogger(DIR_ROOT, name=None, version=setup_name_from_args(args))
    
    #this line of code is served only to change default name of csv file for logging on
    logger.experiment.metrics_file_path = os.path.join( DIR_SETUP, 'logs.csv' )

    checkpoint_logger = ModelCheckpoint(dirpath=DIR_SETUP,
                                        filename='{epoch}_{step}',
                                        save_weights_only=False,
                                        # mode="max", 
                                        monitor=None,
                                        every_n_epochs=1, 
                                        save_top_k=1, ## =1 means save just 1 model and then the last one when training ends
                                        save_on_train_epoch_end=True)
    
    early_stopper = EarlyStopping(monitor='train_loss', patience=2, min_delta=0.02)

    max_epochs = args.max_epochs
    trainer = L.Trainer(
        # fast_dev_run=True, # XXX Enable here for trying scripts
        accelerator="auto",
        strategy="auto",
        num_nodes=1,
        logger=logger,
        default_root_dir=DIR_ROOT,
        callbacks=[checkpoint_logger,early_stopper],
        devices=1,
        max_epochs=max_epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=-1, # to avoid warning in terminal. We want just to track every epoch
        # enable_progress_bar = False # XXX it disables bar progress... useful for HPC
    )
    
    ## Let's start !
    trainer.fit(model=model,
                train_dataloaders=trainDataloader,
                val_dataloaders=validDataloader,
                ckpt_path=ckpt_path,
                )
    
    last_model = last_checkpoint_from_path(DIR_SETUP)
    if last_model is None:
        return

    ckpt_name,ckpt_ext = os.path.splitext( last_model )
    if not '_early_stopping' in ckpt_name:
        new_ckpt_name = ckpt_name + '_early_stopping'
        os.rename(os.path.join(DIR_SETUP,ckpt_name + ckpt_ext),
                os.path.join(DIR_SETUP,new_ckpt_name + ckpt_ext),
                    )
    

if __name__ == '__main__':
    from src.configs import get_parser

    parser = get_parser()

    parser.add_argument('--max_epochs', type=int, default=30, help='Number of epochs for training')

    args, unk = parser.parse_known_args()
    if unk:
        print('Unknown arguments passed !')
        print(*unk)
    
    exit = train(args)

