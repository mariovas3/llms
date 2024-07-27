import torch

from src.metadata import metadata
from src.model.gpt import GPT
from src.model.utils import check_models_params, load_pretrained_weights_


def test_gpt_forward_pass_postnorm_ln():
    d_model = 15
    vocab_size = 11
    gpt = GPT(
        num_layers=2,
        vocab_size=vocab_size,
        context_length=13,
        d_model=d_model,
        num_heads=3,
        ffn_hidden=45,
        activation="gelu",
        dropout=0.1,
        qkv_bias=False,
        norm_first=True,
        pre_norm=False,
    )
    tgt = torch.LongTensor([[2, 4, 1, 7, 0, 0, 0], [3, 5, 8, 1, 2, 0, 0]])
    tgt_key_pad_mask = tgt == 0
    out = gpt(tgt, tgt_key_pad_mask)
    assert out.shape == (*tgt.shape, vocab_size)


def test_gpt_forward_pass_postnorm_noln():
    d_model = 15
    vocab_size = 11
    gpt = GPT(
        num_layers=2,
        vocab_size=vocab_size,
        context_length=13,
        d_model=d_model,
        num_heads=3,
        ffn_hidden=45,
        activation="gelu",
        dropout=0.1,
        qkv_bias=False,
        norm_first=False,
        pre_norm=False,
    )
    tgt = torch.LongTensor([[2, 4, 1, 7, 0, 0, 0], [3, 5, 8, 1, 2, 0, 0]])
    tgt_key_pad_mask = tgt == 0
    out = gpt(tgt, tgt_key_pad_mask)
    assert out.shape == (*tgt.shape, vocab_size)


def test_gpt_forward_pass_prenorm_noln():
    d_model = 15
    vocab_size = 11
    gpt = GPT(
        num_layers=2,
        vocab_size=vocab_size,
        context_length=13,
        d_model=d_model,
        num_heads=3,
        ffn_hidden=45,
        activation="gelu",
        dropout=0.1,
        qkv_bias=False,
        norm_first=False,
        pre_norm=True,
    )
    tgt = torch.LongTensor([[2, 4, 1, 7, 0, 0, 0], [3, 5, 8, 1, 2, 0, 0]])
    tgt_key_pad_mask = tgt == 0
    out = gpt(tgt, tgt_key_pad_mask)
    assert out.shape == (*tgt.shape, vocab_size)


def test_gpt_forward_pass_prenorm_ln():
    d_model = 15
    vocab_size = 11
    gpt = GPT(
        num_layers=2,
        vocab_size=vocab_size,
        context_length=13,
        d_model=d_model,
        num_heads=3,
        ffn_hidden=45,
        activation="gelu",
        dropout=0.1,
        qkv_bias=False,
        norm_first=True,
        pre_norm=True,
    )
    tgt = torch.LongTensor([[2, 4, 1, 7, 0, 0, 0], [3, 5, 8, 1, 2, 0, 0]])
    tgt_key_pad_mask = tgt == 0
    out = gpt(tgt, tgt_key_pad_mask)
    assert out.shape == (*tgt.shape, vocab_size)


def test_weight_loading():
    pretrained_model = "gpt2-medium"
    config = metadata.BASE_CONFIG.copy()
    config.update(metadata.MODEL_CONFIGS[pretrained_model])
    config["ffn_hidden"] = 4 * config["d_model"]
    my_gpt = GPT(**config)
    # load relevant openai weights;
    my_gpt, gpt_hf = load_pretrained_weights_(
        my_gpt, cache_dir=metadata.SAVED_MODELS_PATH, name=pretrained_model
    )
    # check all weights match;
    check_models_params(my_gpt=my_gpt, gpt_hf=gpt_hf)
