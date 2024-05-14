from copy import deepcopy
from math import sqrt
from typing import Literal, Optional

import torch
from torch import nn


class PosEmbed(nn.Module):
    def __init__(self, num_embeds, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeds, embed_dim)

    def forward(self, x):
        return x + self.embed(torch.tensor(range(x.size(-2))))


def get_subsequent_mask(size):
    """
    Returns square boolean mask.

    True values will be ignored in attention.

    The lower triangle is False, the remaining entries are True.
    """
    return torch.triu(torch.ones((size, size)), diagonal=1).bool()


def merge_masks(attn_mask, num_heads, key_padding_mask=None):
    """
    Merge subsequent attention mask with padding mask.

    attn_mask: of shape (Sq, Sk) where Sq is num queries, and
        Sk is num keys.
    num_heads: number of attention heads.
    key_padding_mask: of shape (B, Sk) where B is the batch size
        and Sk is num keys.
    """
    if key_padding_mask is None:
        return attn_mask
    assert attn_mask.ndim == key_padding_mask.ndim == 2
    B, Sk = key_padding_mask.shape
    Sq, Sk = attn_mask.shape
    return attn_mask.expand(B, num_heads, -1, -1) + key_padding_mask.view(
        B, 1, 1, Sk
    ).expand(-1, num_heads, Sq, -1)


def get_clones(module: nn.Module, N):
    return nn.ModuleList([deepcopy(module) for _ in range(N)])


class MultiHeadAtt(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.dk = d_model // num_heads
        assert (
            self.dk * num_heads == d_model
        ), "d_model must be a multiple of num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.Uq = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.Uk = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.Uv = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.Uo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_pad_mask=None):
        B, n, d_model = tgt.shape
        m = memory.size(-2)
        assert d_model == self.d_model
        # tgt is (B, n, d_model)
        Q = (
            self.Uq(tgt).view(B, n, self.num_heads, self.dk).transpose(-3, -2)
        )  # (B, H, n, dk)
        K = (
            self.Uk(memory)
            .transpose(-1, -2)
            .view(B, self.num_heads, self.dk, m)
        )  # (B, H, dk, m)
        V = (
            self.Uv(memory)
            .view(B, m, self.num_heads, self.dk)
            .transpose(-3, -2)
        )  # (B, H, m, dk)

        A = Q @ K / sqrt(self.dk)  # (B, H, n, m)
        if tgt_mask is not None:
            merged_mask = merge_masks(tgt_mask, tgt_key_pad_mask)
            A.masked_fill_(merged_mask, float("-inf"))
        A = self.dropout(torch.softmax(A, -1))
        messages = (A @ V).transpose(-3, -2).contiguous().view(B, n, -1)
        return self.Uo(messages)


class MLP(nn.Module):
    def __init__(
        self, in_dim, ffn_hidden, activation: Literal["gelu", "relu"]
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, ffn_hidden),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(ffn_hidden, in_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResConnection(nn.Module):
    def __init__(self, in_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, residual):
        return self.norm(x + self.dropout(residual))


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_hidden,
        activation: Literal["gelu", "relu"],
        dropout=0.0,
        qkv_bias=False,
    ):
        super().__init__()
        self.mha = MultiHeadAtt(d_model, num_heads, dropout, qkv_bias)
        self.res1 = ResConnection(d_model, dropout)
        self.mlp = MLP(d_model, ffn_hidden, activation)
        self.res2 = ResConnection(d_model, dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_pad_mask=None):
        tgt = self.res1(tgt, self.mha(tgt, memory, tgt_mask, tgt_key_pad_mask))
        return self.res2(tgt, self.mlp(tgt))


class TransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, pre_norm: Optional[nn.Module] = None
    ):
        super().__init__()
        self.layers = get_clones(decoder_layer, num_layers)
        self.pre_norm = pre_norm

    def forward(self, tgt, memory=None, tgt_mask=None, tgt_key_pad_mask=None):
        if self.pre_norm is not None:
            tgt = self.pre_norm(tgt)
        for l in self.layers:
            if memory is None:
                tgt = l(tgt, tgt, tgt_mask, tgt_key_pad_mask)
            else:
                tgt = l(tgt, memory, tgt_mask, tgt_key_pad_mask)
        return tgt