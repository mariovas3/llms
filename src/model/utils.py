from copy import deepcopy
from math import sqrt
from typing import Literal, Optional

import numpy as np
import torch
from torch import nn
from transformers import GPT2Model


class PosEmbed(nn.Module):
    def __init__(self, num_embeds, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeds, embed_dim)

    def forward(self, x):
        return x + self.embed.weight[: x.size(-2), :]


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

        A = Q @ K * (1.0 / sqrt(self.dk))  # (B, H, n, m)
        if tgt_mask is not None:
            merged_mask = merge_masks(
                attn_mask=tgt_mask,
                num_heads=self.num_heads,
                key_padding_mask=tgt_key_pad_mask,
            )
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
            nn.GELU(approximate="tanh") if activation == "gelu" else nn.ReLU(),
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


class PreNormLogic(nn.Module):
    def __init__(self, in_dim, mha, mlp, dropout=0.0):
        super().__init__()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.mha = mha
        self.mlp = mlp
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)

    def forward(self, tgt, tgt_mask=None, tgt_key_pad_mask=None):
        resid = self.norm1(tgt)
        resid = self.mha(resid, resid, tgt_mask, tgt_key_pad_mask)
        tgt = tgt + self.drop1(resid)

        resid = self.norm2(tgt)
        resid = self.mlp(resid)
        return tgt + self.drop2(resid)


class PostNormLogic(nn.Module):
    def __init__(self, in_dim, mha, mlp, dropout=0.0):
        super().__init__()
        self.mha = mha
        self.mlp = mlp
        self.res1 = ResConnection(in_dim, dropout)
        self.res2 = ResConnection(in_dim, dropout)

    def forward(self, tgt, tgt_mask=None, tgt_key_pad_mask=None):
        tgt = self.res1(tgt, self.mha(tgt, tgt, tgt_mask, tgt_key_pad_mask))
        return self.res2(tgt, self.mlp(tgt))


class CausalSALayer(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        ffn_hidden,
        activation: Literal["gelu", "relu"],
        dropout=0.0,
        qkv_bias=False,
        pre_norm=False,
    ):
        super().__init__()
        mha = MultiHeadAtt(d_model, num_heads, dropout, qkv_bias)
        mlp = MLP(d_model, ffn_hidden, activation)
        self.pre_norm = pre_norm
        if pre_norm:
            self.mod = PreNormLogic(d_model, mha, mlp, dropout)
        else:
            self.mod = PostNormLogic(d_model, mha, mlp, dropout)

    def forward(self, tgt, tgt_mask=None, tgt_key_pad_mask=None):
        return self.mod(
            tgt=tgt, tgt_mask=tgt_mask, tgt_key_pad_mask=tgt_key_pad_mask
        )


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_layers,
        norm0: Optional[nn.Module] = None,
        **decoder_layer_kwargs,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [CausalSALayer(**decoder_layer_kwargs) for _ in range(num_layers)]
        )
        self.norm0 = norm0

    def forward(self, tgt, tgt_mask=None, tgt_key_pad_mask=None):
        if self.norm0 is not None:
            tgt = self.norm0(tgt)
        for l in self.layers:
            tgt = l(
                tgt=tgt, tgt_mask=tgt_mask, tgt_key_pad_mask=tgt_key_pad_mask
            )
        return tgt


def assign_first_to_second_(weights1, weights2):
    assert len(weights1) == len(weights2)
    for weight1, weight2 in zip(weights1, weights2):
        assert (
            weight1.shape == weight2.shape
        ), f"{weight1.shape=}, {weight2.shape}"
        weight1.data = weight2.data.detach().clone()


def load_pretrained_weights_(my_gpt, cache_dir, name="gpt2-medium"):
    """
    Load params of OpenAI model to my_gpt inplace.

    Returns my_gpt and the OpenAI model.
    """
    # get the openai model;
    gpt_hf = GPT2Model.from_pretrained(
        f"openai-community/{name}", cache_dir=cache_dir
    )
    # load one-offs;
    assign_first_to_second_(
        (
            my_gpt.embeddings.weight,
            my_gpt.pos_embeddings.embed.weight,
            my_gpt.last_norm.weight,
            my_gpt.last_norm.bias,
            my_gpt.classification_head.weight,
        ),
        (
            gpt_hf.wte.weight,
            gpt_hf.wpe.weight,
            gpt_hf.ln_f.weight,
            gpt_hf.ln_f.bias,
            gpt_hf.wte.weight,
        ),
    )

    # load transformer blocks;
    # note: openai use conv1d weights so will transpose them
    # when loading into my weights;
    # these are the attn and mlp weights (not biases);
    for l1, l2 in zip(gpt_hf.h, my_gpt.decoder.layers):
        # c_attn.weight.shape is (d_model, 3 * d_model)
        q_w, k_w, v_w = np.split(l1.attn.c_attn.weight, 3, -1)
        q_b, k_b, v_b = np.split(l1.attn.c_attn.bias, 3, -1)
        out_w = l1.attn.c_proj.weight
        out_b = l1.attn.c_proj.bias
        d_model = q_w.shape[-1]
        assert 3 * d_model == l1.attn.c_attn.weight.shape[-1]
        assign_first_to_second_(
            (
                l2.mod.mha.Uq.weight,
                l2.mod.mha.Uq.bias,
                l2.mod.mha.Uk.weight,
                l2.mod.mha.Uk.bias,
                l2.mod.mha.Uv.weight,
                l2.mod.mha.Uv.bias,
                l2.mod.mha.Uo.weight,
                l2.mod.mha.Uo.bias,
                l2.mod.norm1.weight,
                l2.mod.norm1.bias,
                l2.mod.norm2.weight,
                l2.mod.norm2.bias,
                l2.mod.mlp.net[0].weight,
                l2.mod.mlp.net[0].bias,
                l2.mod.mlp.net[2].weight,
                l2.mod.mlp.net[2].bias,
            ),
            (
                q_w.t(),  # transpose;
                q_b,
                k_w.t(),  # transpose;
                k_b,
                v_w.t(),  # transpose;
                v_b,
                out_w.t(),  # transpose;
                out_b,
                l1.ln_1.weight,
                l1.ln_1.bias,
                l1.ln_2.weight,
                l1.ln_2.bias,
                l1.mlp.c_fc.weight.t(),  # transpose;
                l1.mlp.c_fc.bias,
                l1.mlp.c_proj.weight.t(),  # transpose;
                l1.mlp.c_proj.bias,
            ),
        )
    return my_gpt, gpt_hf


def check_equal(weights1, weights2):
    for _, (w1, w2) in enumerate(zip(weights1, weights2)):
        assert torch.all(w1.data == w2.data), f"{_}, {w1.shape=}, {w2.shape=}"


def check_models_params(my_gpt, gpt_hf):
    # load one-offs;
    check_equal(
        (
            my_gpt.embeddings.weight,
            my_gpt.pos_embeddings.embed.weight,
            my_gpt.last_norm.weight,
            my_gpt.last_norm.bias,
            my_gpt.classification_head.weight,
        ),
        (
            gpt_hf.wte.weight,
            gpt_hf.wpe.weight,
            gpt_hf.ln_f.weight,
            gpt_hf.ln_f.bias,
            gpt_hf.wte.weight,
        ),
    )

    # load transformer blocks;
    for l1, l2 in zip(gpt_hf.h, my_gpt.decoder.layers):
        # load attention
        # c_attn.weight.shape is (d_model, 3 * d_model)
        q_w, k_w, v_w = np.split(l1.attn.c_attn.weight, 3, -1)
        q_b, k_b, v_b = np.split(l1.attn.c_attn.bias, 3, -1)
        out_w = l1.attn.c_proj.weight
        out_b = l1.attn.c_proj.bias
        d_model = q_w.shape[-1]
        assert 3 * d_model == l1.attn.c_attn.weight.shape[-1]
        check_equal(
            (
                l2.mod.mha.Uq.weight,
                l2.mod.mha.Uq.bias,
                l2.mod.mha.Uk.weight,
                l2.mod.mha.Uk.bias,
                l2.mod.mha.Uv.weight,
                l2.mod.mha.Uv.bias,
                l2.mod.mha.Uo.weight,
                l2.mod.mha.Uo.bias,
                l2.mod.norm1.weight,
                l2.mod.norm1.bias,
                l2.mod.norm2.weight,
                l2.mod.norm2.bias,
                l2.mod.mlp.net[0].weight,
                l2.mod.mlp.net[0].bias,
                l2.mod.mlp.net[2].weight,
                l2.mod.mlp.net[2].bias,
            ),
            (
                q_w.t(),  # transpose;
                q_b,
                k_w.t(),  # transpose;
                k_b,
                v_w.t(),  # transpose;
                v_b,
                out_w.t(),  # transpose;
                out_b,
                l1.ln_1.weight,
                l1.ln_1.bias,
                l1.ln_2.weight,
                l1.ln_2.bias,
                l1.mlp.c_fc.weight.t(),  # transpose;
                l1.mlp.c_fc.bias,
                l1.mlp.c_proj.weight.t(),  # transpose;
                l1.mlp.c_proj.bias,
            ),
        )
