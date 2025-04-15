import math
import numpy as np
from typing import Any

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# PyTorch Lightning
import lightning as L

from .components import PositionalEncoding, TransformerEncoder, CosineWarmupScheduler
from ..configs.constants import BOS, EOS, PAD, UNK

class TransformerModel(L.LightningModule):
    """Documentation to write."""

    def __init__(
        self,
        input_dim, # = output_dim (vocabulary size)
        model_dim=100, # embedding dimension
        num_heads=2,
        num_layers=2,
        batch_size=512,
        lr=1e-3,
        max_len=150,
        dropout=0.5,
        vocab = {PAD:0, UNK:1, BOS:2, EOS:3} ## default for vocabulary
                ):

        super().__init__()
        
        self.setup_memory(vocab)
        self.save_hyperparameters(ignore='vocab')
        self.setup_model()

    def setup_memory(self, vocab):
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.padding_idx = vocab[PAD]
        self.bos_idx = vocab[BOS]
        self.eos_idx = vocab[EOS]
        self.temperature = 1.0


    def setup_model(self):
        self.src_mask = None

        self.input_emb = nn.Embedding(self.hparams.input_dim, self.hparams.model_dim, padding_idx=self.padding_idx)
        self.pos_encoder = PositionalEncoding(self.hparams.model_dim, self.hparams.dropout, self.hparams.max_len)
        self.encoder = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            model_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
            # batch_first=True,
        )

        self.decoder = nn.Linear(self.hparams.model_dim, self.hparams.input_dim)
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz))).to(self.device)

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.input_emb.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, has_mask=True):
        """
        Args:
            x: Input features of shape [SeqLen, Batch]
            mask: Mask to apply on the attention outputs (optional)
            add_positional_encoding: If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """

        if has_mask:
            if self.src_mask is None or self.src_mask.size(0) != len(x):
                mask = self._generate_square_subsequent_mask(len(x))
                self.src_mask = mask
        else:
            self.src_mask = None

        x = self.input_emb(x) * math.sqrt(self.hparams.input_dim)
        # if add_positional_encoding:
        x = self.pos_encoder(x) ## prima era condizionato da un booleano
        x = self.encoder(x, mask=self.src_mask)
        x = self.decoder(x)
        return F.log_softmax(x, dim=-1)

    # @torch.no_grad()
    # def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
    #     """Function for extracting the attention matrices of the whole Transformer for a single batch.

    #     Input arguments same as the forward pass.
    #     """
    #     # x = self.input_net(x)
    #     x = self.input_emb(x) * math.sqrt(self.hparams.input_dim)
    #     if add_positional_encoding:
    #         x = self.pos_encoder(x)
    #     # attention_maps = self.transformer.get_attention_maps(x, mask=mask)
    #     attention_maps = 
    #     return attention_maps

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    ## default fucntion for loss calculation
    def _calculate_loss(self, batch):
        output = self.forward(batch[:-1]) ## skipping the last token
        loss = F.cross_entropy(
            output.view(-1, self.hparams.input_dim), 
            batch[1:].view(-1), ## skipping the first token
            ignore_index=self.padding_idx,
            )

        return loss

    ## this method takes into account end of training and validation (if provided) epoch.
    ## here is usefull for loggin metrics
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        loss_epoch = torch.stack([out['loss'] for out in self.training_step_outputs]).mean()

        self.log(f"train_loss", loss_epoch, on_step=False, on_epoch=True, prog_bar=False, logger=True)
   
        self.training_step_outputs.clear()

        if self.validation_step_outputs:
            loss_epoch = torch.stack([out['loss'] for out in self.validation_step_outputs]).mean()
            self.log(f"val_loss", loss_epoch, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            self.validation_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        loss=self._calculate_loss(batch)

        self.log(f"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.hparams.batch_size)

        outputs = dict(loss=loss)

        self.training_step_outputs.append(outputs)
        return outputs

    def validation_step(self, batch, batch_idx):
        loss=self._calculate_loss(batch)

        self.log(f"val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.hparams.batch_size)

        outputs = dict(loss=loss)
        self.validation_step_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        raise NotImplementedError

class TransformerCosine(TransformerModel):
    def __init__(self,
        input_dim, # = output_dim (vocabulary size)
        model_dim=100, # embedding dimension
        num_heads=2,
        num_layers=2,
        batch_size=512,
        lr=1e-3,
        max_len=150,
        dropout=0.5,
        warmup=100,
        max_iters=1000,
        vocab = {PAD:0, UNK:1, BOS:2, EOS:3} ## default for vocabulary
                 ):
        L.LightningModule.__init__(self)
        
        self.setup_memory(vocab)
        self.save_hyperparameters(ignore='vocab')
        self.setup_model()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
        self.lr_scheduler = lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )

        lr_scheduler_config = {
        "scheduler": lr_scheduler,
        "interval": "step",
        "frequency": 1,
        # "monitor": "lr_config",
        "strict": True,
        # "name": 'LearningRateMonitor',
        }

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, batch, batch_idx):
        outputs = super().training_step(batch, batch_idx)

        lr = self.lr_scheduler.get_last_lr()[0]
        self.log(f"lr", lr, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        self.training_step_outputs[-1] = outputs | {'lr':lr}
        return outputs
    
    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        last_lr = self.training_step_outputs[-1]['lr']
        self.log(f"lr", last_lr, on_step=False, on_epoch=True, prog_bar=False, logger=True)

class TransformerReactions(L.LightningModule):
    def __init__(
        self,
        input_dim, # = output_dim (vocabulary size)
        model_dim=100, # embedding dimension
        num_heads=2,
        num_layers=2,
        batch_size=512,
        lr=1e-3,
        max_len=150,
        dropout=0.5,
        vocab = {PAD:0, UNK:1, BOS:2, EOS:3} ## default by vocabulary
                ):

        super().__init__()
        
        self.setup_memory(vocab)
        self.save_hyperparameters(ignore='vocab')
        self.setup_model()

    def setup_memory(self, vocab):
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.padding_idx = vocab[PAD]
        self.bos_idx = vocab[BOS]
        self.eos_idx = vocab[EOS]
        self.temperature = 1.0

    def setup_model(self):
        self.tgt_mask = None

        self.input_emb = nn.Embedding(self.hparams.input_dim,
                                      self.hparams.model_dim,
                                      padding_idx=self.padding_idx
                                      )

        self.pos_encoder = PositionalEncoding(self.hparams.model_dim, 
                                              self.hparams.dropout, 
                                              self.hparams.max_len
                                              )

        self.transformer = nn.Transformer(d_model=self.hparams.model_dim, 
                                          nhead=self.hparams.num_heads, 
                                          dropout=self.hparams.dropout,
                                          num_encoder_layers=self.hparams.num_layers,
                                          num_decoder_layers=self.hparams.num_layers,
                                        )

        self.fc = nn.Linear(self.hparams.model_dim,
                            self.hparams.input_dim
                            )
        # self.init_weights()

    def make_pad_mask(self, x):
        return (x.transpose(0, 1) == self.padding_idx).to(self.device)
        # (N, src_len)

    def forward(self, src, tgt):

        if self.tgt_mask is None or self.tgt_mask.size(0) != len(tgt):
            mask = self._generate_square_subsequent_mask(len(tgt))
            self.tgt_mask = mask

        # src_pad=self.make_pad_mask(src)
        # tgt_pad=self.make_pad_mask(tgt)

        src = self.input_emb(src)
        tgt = self.input_emb(tgt)

        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        z = self.transformer(
                    src=src, tgt=tgt, tgt_mask=self.tgt_mask, 
                    # src_key_padding_mask=src_pad, tgt_key_padding_mask=tgt_pad
                       )

        z = self.fc(z)

        return z

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def _generate_square_subsequent_mask(self, sz):
        return torch.log(torch.tril(torch.ones(sz,sz))).to(self.device)

    def _calculate_loss(self, batch):
        src, tgt = batch

        output = self.forward(src, tgt[:-1])

        output = output.reshape(-1, output.shape[2])
        tgt = tgt[1:].reshape(-1) ## skipping the first token

        loss = F.cross_entropy( output, tgt,
            ignore_index=self.padding_idx,
            )

        return loss

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        loss_epoch = torch.stack([out['loss'] for out in self.training_step_outputs]).mean()

        self.log(f"train_loss", loss_epoch, on_step=False, on_epoch=True, prog_bar=False, logger=True)
   
        self.training_step_outputs.clear()

        if self.validation_step_outputs:
            loss_epoch = torch.stack([out['loss'] for out in self.validation_step_outputs]).mean()
            self.log(f"val_loss", loss_epoch, on_step=False, on_epoch=True, prog_bar=False, logger=True)

            self.validation_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        loss=self._calculate_loss(batch)

        self.log(f"train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.hparams.batch_size)

        outputs = dict(loss=loss)

        self.training_step_outputs.append(outputs)
        return outputs

    def validation_step(self, batch, batch_idx):
        loss=self._calculate_loss(batch)

        self.log(f"val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=False, batch_size=self.hparams.batch_size)

        outputs = dict(loss=loss)
        self.validation_step_outputs.append(outputs)
        return outputs

    def predict_step(self, batch, batch_idx):
        src, tgt = batch

        k = self.k

        batch_len, batch_size = tgt.shape

        tgt_pred = torch.full( (1, batch_size, k), self.bos_idx, device=self.device)


        output = self.forward( src, tgt_pred[:,:,0] )[-1] # (batch, tokens) - last dim is 0 cause of first output is only top-1 prediction
        weigths = F.log_softmax(output, dim=-1) ## applying on vocabulary dimension

        logits_cumulative, top_indexes =  weigths.topk(k=k) ## (batch, k)

        logits_pred = logits_cumulative.unsqueeze(0)
        tgt_pred = torch.cat([ tgt_pred, top_indexes.unsqueeze(0) ])

        # mask_eos = np.zeros((batch_size,k), dtype=bool)
        mask_stop = torch.zeros( (batch_size,k), dtype=torch.bool, device=self.device)

        position=2
        while not mask_stop.all() and position < batch_len:
            # Sum along k dimension of mask to recognize how many preds need yet for each actual src/tgt
            k_missing = (~mask_stop).sum(dim=1) ## rows full of False became integer = 5

            tgt_last = torch.full( (batch_size, k), self.padding_idx, device=self.device)
            logits_last = torch.zeros( (batch_size, k), device=self.device)

            for sub_k in k_missing.unique() :
                if sub_k == 0:
                    continue

                mask_b = k_missing == sub_k # (B,) recognize batch idx

                mask_bk = mask_b.unsqueeze(1) & ~mask_stop # (B,k) recognize beam nodes

                sub_batch = mask_b.sum() # size of sub batch

                sub_tgt_extended = tgt_pred[:, mask_bk ]
                sub_tgt = sub_tgt_extended.view(position,sub_batch,sub_k)

                sub_cumulative = logits_cumulative[mask_bk].view(sub_batch,sub_k)

                sub_src_repeated = src[:,mask_b].repeat_interleave(sub_k, dim=1) # (L,sub_batch * sub_k)

                output = self.forward( sub_src_repeated, sub_tgt_extended) [-1] ## (sub_batch*sub_k, tokens)
                weigths = F.log_softmax(output, dim=-1) ## applying on vocabulary dimension (sub_batch*sub_k, tokens)

                top_weigths, top_indexes =  weigths.topk(k=sub_k, sorted=True) ## (sub_batch*sub_k, sub_k)

                top_logits = top_weigths.clone().view(sub_batch, sub_k * sub_k) # raw logits to be stored for the final output
                top_weigths += sub_cumulative.view(sub_batch*sub_k,1) # subcumulative and the relative sum

                top_weigths = top_weigths.view(sub_batch, sub_k * sub_k)
                top_indexes = top_indexes.view(sub_batch, sub_k * sub_k)

                _, topk_sub_batch_idxs = top_weigths.topk(k=sub_k, dim=-1, sorted=True)

                # FIXME i don't like this way to select different columns straight of rows
                rows_idxs = torch.arange(sub_batch, device=self.device).unsqueeze(1)

                next_tokens = top_indexes[rows_idxs,topk_sub_batch_idxs] # shape = (sub_batch, sub_k)
                next_weigths =  top_weigths[rows_idxs,topk_sub_batch_idxs] # shape = (sub_batch, sub_k)
                next_logits = top_logits[rows_idxs,topk_sub_batch_idxs] # shape = (sub_batch, sub_k)

                tgt_last[mask_bk] = next_tokens.view(sub_batch*sub_k)
                logits_cumulative[mask_bk] = next_weigths.view(sub_batch*sub_k)
                logits_last[mask_bk] = next_logits.view(sub_batch*sub_k)

                # FIXME i don't like this way to select different columns straight of rows
                rows_idxs = torch.arange(position, device=self.device).unsqueeze(1).unsqueeze(2)
                cols_idxs = torch.arange(sub_batch, device=self.device).unsqueeze(0).unsqueeze(2)
                topk_sub_batch_idxs = topk_sub_batch_idxs // sub_k
                sub_tgt = sub_tgt[rows_idxs,cols_idxs,topk_sub_batch_idxs.unsqueeze(0)]

                tgt_pred[:, mask_bk] = sub_tgt.view(position, sub_batch * sub_k)

                mask_eos = next_tokens == self.eos_idx
                if not mask_eos.any():
                    continue

                bk_idxs = torch.argwhere(mask_bk) ## = rows (sub_batch), columns (sub_k)
                bk_eos_idxs = bk_idxs[mask_eos.flatten()]

                mask_stop[*bk_eos_idxs.T]=True

            tgt_pred = torch.cat([ tgt_pred, tgt_last.unsqueeze(0) ])
            logits_pred = torch.cat([ logits_pred, logits_last.unsqueeze(0) ])

            position+=1

        # return tgt_pred
        return {'predictions':tgt_pred, 'logits_cum':logits_cumulative,'logits':logits_pred}