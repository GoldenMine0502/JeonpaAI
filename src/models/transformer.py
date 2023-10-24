import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
import numpy as np


class Transformer(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(Transformer, self).__init__()
        self.pred_len = configs.model.pred_len
        self.output_attention = configs.model.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.model.enc_in, configs.model.d_model, configs.model.embed, configs.model.freq,
                                           configs.model.dropout)
        self.dec_embedding = DataEmbedding(configs.model.dec_in, configs.model.d_model, configs.model.embed, configs.model.freq,
                                           configs.model.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.model.factor, attention_dropout=configs.model.dropout,
                                      output_attention=configs.model.output_attention), configs.model.d_model, configs.model.n_heads),
                    configs.model.d_model,
                    configs.model.d_ff,
                    dropout=configs.model.dropout,
                    activation=configs.model.activation
                ) for l in range(configs.model.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.model.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.model.factor, attention_dropout=configs.model.dropout, output_attention=False),
                        configs.model.d_model, configs.model.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.model.factor, attention_dropout=configs.model.dropout, output_attention=False),
                        configs.model.d_model, configs.model.n_heads),
                    configs.model.d_model,
                    configs.model.d_ff,
                    dropout=configs.model.dropout,
                    activation=configs.model.activation,
                )
                for l in range(configs.model.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.model.d_model),
            projection=nn.Linear(configs.model.d_model, configs.model.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]