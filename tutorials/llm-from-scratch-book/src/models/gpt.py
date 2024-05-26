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
        pre_norm = nn.LayerNorm(d_model) if pre_norm else None
        self.decoder = utils.TransformerDecoder(
            decoder_layer, num_layers, pre_norm
        )
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.pos_embeddings = utils.PosEmbed(context_length, d_model)
        self.drop_embed = nn.Dropout(dropout)
        self.mask = utils.get_subsequent_mask(context_length)
        self.classification_head = nn.Linear(d_model, vocab_size, bias=False)

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

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--vocab_size",
            type=int,
            default=50_257,
        )
        parser.add_argument(
            "--context_length",
            type=int,
            default=1024,
        )
        parser.add_argument(
            "--d_model",
            type=int,
            default=768,
        )
        parser.add_argument(
            "--num_heads",
            type=int,
            default=12,
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=12,
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
        )
        parser.add_argument(
            "--qkv_bias",
            action="store_true",
        )
        parser.add_argument(
            "--ffn_hidden",
            type=int,
            default=768 * 4,
        )
        parser.add_argument("--activation", type=str, default="gelu")
        parser.add_argument("--pre_norm", action="store_true")
        return parser


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = GPT.add_to_argparse(parser)
    parser = parser.parse_args()
    args = vars(parser)
    model = GPT(**args)
    n_params = sum(p.numel() for p in model.parameters())
    less_classification_head = n_params - sum(
        p.numel() for p in model.classification_head.parameters()
    )
    print(
        f"n_params: {n_params / 1e6}M\nless classification head: {less_classification_head / 1e6}M"
    )
    block_weight = 0
    for key, val in model.named_parameters():
        if "layers" in key:
            idx = int(key.split(".")[2])
            if idx > 0:
                continue
            block_weight += val.numel()
        print(f"{key}: {val.shape}")
    print(f"block_weight={block_weight / 1e6}M")
