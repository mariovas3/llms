from math import sqrt

import torch
from torch import nn

from src.model.utils import get_clones, merge_masks


class SingleHeadAtt(nn.Module):
    """
    Outputs A @ V.

    Only for testing purposes.
    """

    def __init__(self, d_model, dk, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.Uq = nn.Linear(d_model, dk, bias=qkv_bias)
        self.Uk = nn.Linear(d_model, dk, bias=qkv_bias)
        self.Uv = nn.Linear(d_model, dk, bias=qkv_bias)
        self.d_model = d_model
        self.dk = dk
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_pad_mask=None):
        Q, K, V = self.Uq(tgt), self.Uk(memory), self.Uv(memory)
        A = Q @ K.transpose(-1, -2) / sqrt(self.dk)
        if tgt_mask is not None:
            merged_mask = merge_masks(
                attn_mask=tgt_mask,
                num_heads=1,
                key_padding_mask=tgt_key_pad_mask,
            )
            A.masked_fill_(merged_mask, float("-inf"))
        A = self.dropout(torch.softmax(A, -1))
        return A @ V


class NaiveMHA(nn.Module):
    def __init__(self, num_heads, d_model, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.dk = d_model // num_heads
        assert (
            self.dk * num_heads == d_model
        ), "d_model must be a multiple of num_heads"
        self.heads = get_clones(
            SingleHeadAtt(d_model, self.dk, dropout, qkv_bias), num_heads
        )
        self.Uo = nn.Linear(d_model, d_model)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_pad_mask=None):
        return self.Uo(
            torch.cat(
                [
                    h(tgt, memory, tgt_mask, tgt_key_pad_mask)
                    for h in self.heads
                ],
                -1,
            )
        )


class MultiHeadAttSlower(nn.Module):
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
            .view(B, m, self.num_heads, self.dk)
            .transpose(-3, -2)
        )  # (B, H, dk, m)
        V = (
            self.Uv(memory)
            .view(B, m, self.num_heads, self.dk)
            .transpose(-3, -2)
        )  # (B, H, m, dk)

        A = Q @ K.transpose(-1, -2) / sqrt(self.dk)  # (B, H, n, m)
        if tgt_mask is not None:
            merged_mask = merge_masks(
                attn_mask=tgt_mask,
                num_heads=self.num_heads,
                key_padding_mask=tgt_key_pad_mask,
            )
            A = A.masked_fill(merged_mask, float("-inf"))
        A = self.dropout(torch.softmax(A, -1))
        messages = (A @ V).transpose(-3, -2).contiguous().view(B, n, -1)
        return self.Uo(messages)


def copy_mha_weights(naive_mha, parallel_mha):
    # make sure I have the same weights;
    Uq, Uk, Uv = [], [], []
    Uq_b, Uk_b, Uv_b = [], [], []
    for m in naive_mha.heads:
        Uq.append(m.Uq.weight.data)
        if m.Uq.bias is not None:
            Uq_b.append(m.Uq.bias.data)
        Uk.append(m.Uk.weight.data)
        if m.Uk.bias is not None:
            Uk_b.append(m.Uk.bias.data)
        Uv.append(m.Uv.weight.data)
        if m.Uv.bias is not None:
            Uv_b.append(m.Uv.bias.data)
    parallel_mha.Uo.weight.data = naive_mha.Uo.weight.data
    parallel_mha.Uo.bias.data = naive_mha.Uo.bias.data

    # in torch, nn.Linear(d_model, dk).weight.data.shape is (dk, d_model)!!!
    # so concat along first dim to get (d_model, d_model);
    parallel_mha.Uq.weight.data = torch.cat(Uq, 0)
    if Uq_b:
        parallel_mha.Uq.bias.data = torch.cat(Uq_b, -1)
    parallel_mha.Uk.weight.data = torch.cat(Uk, 0)
    if Uk_b:
        parallel_mha.Uk.bias.data = torch.cat(Uk_b, -1)
    parallel_mha.Uv.weight.data = torch.cat(Uv, 0)
    if Uv_b:
        parallel_mha.Uv.bias.data = torch.cat(Uv_b, -1)
