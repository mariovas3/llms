from typing import Literal

from src.models import utils
from torch import nn


class GPT(nn.Module):
    def __init__(
        self,
        num_layers,
        pre_norm,
        vocab_size,
        context_length,
        d_model,
        num_heads,
        ffn_hidden,
        activation: Literal["gelu", "relu"],
        dropout=0.0,
        qkv_bias=False,
    ):
        super().__init__()
        decoder_layer = utils.TransformerDecoderLayer(
            d_model, num_heads, ffn_hidden, activation, dropout, qkv_bias
        )
        self.decoder = utils.TransformerDecoder(
            decoder_layer, num_layers, pre_norm
        )
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = utils.PosEmbed(context_length, d_model)
        self.drop_embed = nn.Dropout(dropout)
        self.mask = utils.get_subsequent_mask(context_length)
        self.classification_head = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, tgt_key_pad_mask=None, inference=False):
        tgt = self.drop_embed(self.pos_embeddings(self.embeddings(tgt)))
        tgt_mask = self.mask[: tgt.size(-2), : tgt.size(-2)]
        tgt = self.decoder(
            tgt, tgt_mask=tgt_mask, tgt_key_pad_mask=tgt_key_pad_mask
        )
        if inference:
            # predict token after last given token;
            # returns (B, vocab_size)
            return self.classification_head(tgt[:, -1, :])
        # returns (B, seq_len, vocab_size)
        return self.classification_head(tgt)
