import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp

class AutoFormer(nn.Module):
    """
    Autoformer is the first method to achieve the series-wise connection,
    with inherent O(LlogL) complexity
    """

    def __init__(self, configs):
        super(AutoFormer, self).__init__()
        self.seq_len = configs.model.seq_len  # int
        self.label_len = configs.model.label_len  # int
        self.pred_len = configs.model.pred_len  # int
        self.output_attention = configs.model.output_attention  # bool

        # Decomp
        kernel_size = configs.model.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(
            configs.model.enc_in,
            configs.model.d_model,
            configs.model.embed,
            configs.model.freq,
            configs.model.dropout)

        self.dec_embedding = DataEmbedding_wo_pos(
            configs.model.dec_in,
            configs.model.d_model,
            configs.model.embed,
            configs.model.freq,
            configs.model.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.model.factor, attention_dropout=configs.model.dropout,
                                        output_attention=configs.model.output_attention),
                        configs.model.d_model, configs.model.n_heads),
                    configs.model.d_model,
                    configs.model.d_ff,
                    moving_avg=configs.model.moving_avg,
                    dropout=configs.model.dropout,
                    activation=configs.model.activation
                ) for l in range(configs.model.e_layers)
            ],
            norm_layer=my_Layernorm(configs.model.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, configs.model.factor, attention_dropout=configs.model.dropout,
                                        output_attention=False),
                        configs.model.d_model, configs.model.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.model.factor, attention_dropout=configs.model.dropout,
                                        output_attention=False),
                        configs.model.d_model, configs.model.n_heads),
                    configs.model.d_model,
                    configs.model.c_out,
                    configs.model.d_ff,
                    moving_avg=configs.model.moving_avg,
                    dropout=configs.model.dropout,
                    activation=configs.model.activation,
                )
                for l in range(configs.model.d_layers)
            ],
            norm_layer=my_Layernorm(configs.model.d_model),
            projection=nn.Linear(configs.model.d_model, configs.model.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # print(x_enc.shape, x_dec.shape, x_mark_enc.shape, x_mark_dec.shape)
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)
        seasonal_init, trend_init = self.decomp(x_enc)
        # print("decomp init:", trend_init.shape, seasonal_init.shape)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1) # label_len + pred_len
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # print("decoder input:", trend_init.shape, seasonal_init.shape)
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # print("enc:", enc_out.shape, seasonal_init.shape, x_mark_dec.shape)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
