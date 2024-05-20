import tests.utils as tu
import torch
from src.models import utils


def test_pos_embeds(batch_3_5_7):
    pos_embed = utils.PosEmbed(batch_3_5_7.size(-2) + 5, batch_3_5_7.size(-1))
    out = pos_embed(batch_3_5_7)
    assert out.shape == batch_3_5_7.shape


def test_parallel_mha_no_bias():
    tgt = torch.randn(3, 5, 8)
    memory = torch.randn(3, 7, 8)
    qkv_bias = False
    naive_mha = tu.NaiveMHA(num_heads=2, d_model=8, qkv_bias=qkv_bias)
    tgt_mask = utils.get_subsequent_mask(7)[: tgt.size(-2), : memory.size(-2)]
    out1 = naive_mha(tgt, memory, tgt_mask)
    assert out1.shape == tgt.shape
    mha = utils.MultiHeadAtt(d_model=8, num_heads=2, qkv_bias=qkv_bias)
    # make sure I have the same weights;
    tu.copy_mha_weights(naive_mha, mha)
    # predict;
    out2 = mha(tgt, memory, tgt_mask)
    # test all close;
    assert torch.allclose(out1, out2)


def test_parallel_mha_with_bias():
    tgt = torch.randn(3, 5, 8)
    memory = torch.randn(3, 7, 8)
    qkv_bias = True
    naive_mha = tu.NaiveMHA(num_heads=2, d_model=8, qkv_bias=qkv_bias)
    tgt_mask = utils.get_subsequent_mask(7)[: tgt.size(-2), : memory.size(-2)]
    out1 = naive_mha(tgt, memory, tgt_mask)
    assert out1.shape == tgt.shape
    mha = utils.MultiHeadAtt(d_model=8, num_heads=2, qkv_bias=qkv_bias)
    # make sure I have the same weights;
    tu.copy_mha_weights(naive_mha, mha)
    # predict
    out2 = mha(tgt, memory, tgt_mask)
    # test all close;
    assert torch.allclose(out1, out2)


def test_parallel_mha():
    tgt = torch.randn(3, 5, 8)
    memory = torch.randn(3, 7, 8)
    qkv_bias = True
    tgt_mask = utils.get_subsequent_mask(7)[: tgt.size(-2), : memory.size(-2)]

    naive_mha = tu.MultiHeadAttSlower(
        d_model=8, num_heads=2, qkv_bias=qkv_bias
    )
    out1 = naive_mha(tgt, memory, tgt_mask)
    assert out1.shape == tgt.shape
    mha = utils.MultiHeadAtt(d_model=8, num_heads=2, qkv_bias=qkv_bias)
    # make sure I have the same weights;
    for key, param in mha.named_parameters():
        param.data = naive_mha.get_parameter(key).data
    # predict
    out2 = mha(tgt, memory, tgt_mask)
    # test all close;
    assert torch.allclose(out1, out2)
