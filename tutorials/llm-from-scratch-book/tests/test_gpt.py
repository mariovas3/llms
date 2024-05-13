import torch
from src.models.gpt import GPT


def test_gpt_forward_pass_ln_drop():
    d_model = 15
    vocab_size = 11
    gpt = GPT(
        num_layers=2,
        pre_norm=torch.nn.LayerNorm(d_model),
        vocab_size=vocab_size,
        context_length=13,
        d_model=d_model,
        num_heads=3,
        ffn_hidden=45,
        activation="gelu",
        dropout=0.1,
        qkv_bias=False,
    )
    tgt = torch.LongTensor([[2, 4, 1, 7, 0, 0, 0], [3, 5, 8, 1, 2, 0, 0]])
    tgt_key_pad_mask = tgt == 0
    out = gpt(tgt, tgt_key_pad_mask)
    assert out.shape == (*tgt.shape, vocab_size)


def test_gpt_forward_pass_noln_drop():
    d_model = 15
    vocab_size = 11
    gpt = GPT(
        num_layers=2,
        pre_norm=None,
        vocab_size=vocab_size,
        context_length=13,
        d_model=d_model,
        num_heads=3,
        ffn_hidden=45,
        activation="gelu",
        dropout=0.1,
        qkv_bias=False,
    )
    tgt = torch.LongTensor([[2, 4, 1, 7, 0, 0, 0], [3, 5, 8, 1, 2, 0, 0]])
    tgt_key_pad_mask = tgt == 0
    out = gpt(tgt, tgt_key_pad_mask)
    assert out.shape == (*tgt.shape, vocab_size)


def test_gpt_forward_pass_noln_nodrop():
    d_model = 15
    vocab_size = 11
    gpt = GPT(
        num_layers=2,
        pre_norm=torch.nn.LayerNorm(d_model),
        vocab_size=vocab_size,
        context_length=13,
        d_model=d_model,
        num_heads=3,
        ffn_hidden=45,
        activation="gelu",
        dropout=0.0,
        qkv_bias=False,
    )
    tgt = torch.LongTensor([[2, 4, 1, 7, 0, 0, 0], [3, 5, 8, 1, 2, 0, 0]])
    tgt_key_pad_mask = tgt == 0
    out = gpt(tgt, tgt_key_pad_mask)
    assert out.shape == (*tgt.shape, vocab_size)
