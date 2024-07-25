from typing import Literal

from torch import nn

from src.model import utils


# the defaults correspond to gpt2 small 124M params.
class GPT(nn.Module):
    def __init__(
        self,
        num_layers=12,
        vocab_size=50257,
        context_length=1024,
        d_model=768,
        num_heads=12,
        ffn_hidden=4 * 768,
        activation: Literal["gelu", "relu"] = "gelu",
        dropout=0.1,
        qkv_bias=True,
        norm_first=True,
        pre_norm=True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        # need to norm again before classification head
        # if pre norm; see https://arxiv.org/pdf/2002.04745
        if pre_norm:
            self.last_norm = nn.LayerNorm(d_model)

        # see if should start with layer_norm(tgt) instead of tgt;
        norm0 = nn.LayerNorm(d_model) if norm_first else None
        self.decoder = utils.TransformerDecoder(
            num_layers=num_layers,
            norm0=norm0,
            d_model=d_model,
            num_heads=num_heads,
            ffn_hidden=ffn_hidden,
            activation=activation,
            dropout=dropout,
            qkv_bias=qkv_bias,
            pre_norm=pre_norm,
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
        if self.pre_norm:
            tgt = self.last_norm(tgt)
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
        parser.add_argument(
            "--pre_norm",
            action="store_true",
            help="Pre norming or post norming.",
        )
        parser.add_argument(
            "--norm_first",
            action="store_true",
            help="before any forward passes, do tgt = layer_norm(tgt).",
        )
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
        f"n_params: {n_params / 1e6}M\nless "
        f"classification head: {less_classification_head / 1e6}M"
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
