import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import ReformerLayer
from layers.Embed import DataEmbedding
import numpy as np


class Reformer(nn.Module):
    """
    Reformer with O(LlogL) complexity
    - It is notable that Reformer is not proposed for time series forecasting, in that it cannot accomplish the cross attention.
    - Here is only one adaption in BERT-style, other possible implementations can also be acceptable.
    - The hyper-parameters, such as bucket_size and n_hashes, need to be further tuned.
    The official repo of Reformer (https://github.com/lucidrains/reformer-pytorch) can be very helpful, if you have any questiones.
    """

    def __init__(self, configs):
        super(Reformer, self).__init__()
        self.pred_len = configs.model.pred_len
        self.pred_len = configs.model.pred_len
        self.output_attention = configs.model.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.model.enc_in, configs.model.d_model, configs.model.embed, configs.model.freq,
                                           configs.model.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ReformerLayer(None, configs.model.d_model, configs.model.n_heads, bucket_size=configs.model.bucket_size,
                                  n_hashes=configs.model.n_hashes),
                    configs.model.d_model,
                    configs.model.d_ff,
                    dropout=configs.model.dropout,
                    activation=configs.model.activation
                ) for l in range(configs.model.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.model.d_model)
        )
        self.projection = nn.Linear(configs.model.d_model, configs.model.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # add placeholder
        x_enc = torch.cat([x_enc, x_dec[:, -self.pred_len:, :]], dim=1)
        x_mark_enc = torch.cat([x_mark_enc, x_mark_dec[:, -self.pred_len:, :]], dim=1)
        # Reformer: encoder only
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out[:, -self.pred_len:, :], attns
        else:
            return enc_out[:, -self.pred_len:, :]  # [B, L, D]