from typing import Literal, Optional

import torch
from lightning import LightningModule
from torch import nn

from src.metadata import metadata
from src.model import lora_utils, utils
from src.model.gpt import GPT


class LitGPT(LightningModule):
    def __init__(
        self,
        lora_rank: int = 8,
        lora_alpha: float = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        do_ffn: bool = False,
        do_lora: bool = True,
        num_layers=12,
        vocab_size=50257,
        context_length=1024,
        d_model=768,
        num_heads=12,
        ffn_hidden=4 * 768,
        activation: Literal["gelu", "relu"] = "gelu",
        dropout=0.1,
        qkv_bias=True,
        norm_first=False,
        pre_norm=True,
        from_pretrained_model: Optional[
            Literal["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
        ] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        if from_pretrained_model:
            config = metadata.BASE_CONFIG.copy()
            config.update(metadata.MODEL_CONFIGS[from_pretrained_model])
            config["ffn_hidden"] = 4 * config["d_model"]
            self.model = GPT(**config)
            # load relevant openai weights;
            utils.load_pretrained_weights_(
                self.model,
                cache_dir=metadata.SAVED_MODELS_PATH,
                name=from_pretrained_model,
            )
        else:
            self.model = GPT(
                num_layers=num_layers,
                vocab_size=vocab_size,
                context_length=context_length,
                d_model=d_model,
                num_heads=num_heads,
                ffn_hidden=ffn_hidden,
                activation=activation,
                dropout=dropout,
                qkv_bias=qkv_bias,
                norm_first=norm_first,
                pre_norm=pre_norm,
            )

        self.do_lora = do_lora
        if do_lora:
            # detach model params from grad tracking;
            self.model.requires_grad_(False)
            # get lora weights;
            self.lora_module_list = lora_utils.init_lora_module_list_qv(
                self.model, rank=lora_rank, alpha=lora_alpha, do_ffn=do_ffn
            )
            # load lora weights;
            lora_utils.load_lora_layers_qv_(
                self.model, self.lora_module_list, do_ffn=do_ffn
            )

    def configure_optimizers(self):
        if self.do_lora:
            return torch.optim.Adam(
                self.lora_module_list.parameters(), lr=self.hparams["lr"]
            )
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def forward(self, tokens, tgt_key_pad_mask=None):
        """Returns logits of shape (B, S, vocab_size)."""
        return self.model(tokens, tgt_key_pad_mask=tgt_key_pad_mask)

    def _get_loss(self, logits, targets):
        # logits shape is (B, S, Vocab) -> need to transpose last 2 dims;
        # targets shape is (B, S)
        return nn.functional.cross_entropy(
            input=logits.transpose(-1, -2),
            target=targets,
            ignore_index=-100,
            reduction="mean",
        )

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        tokens, targets, att_pad_masks = batch
        logits = self(tokens, tgt_key_pad_mask=att_pad_masks)
        loss = self._get_loss(logits, targets)

        # log to logger;
        self.log_dict(
            {
                "training/loss": loss.item(),
                "training/tokens_processed": tokens.numel(),
            },
            logger=True,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        tokens, targets, att_pad_masks = batch
        logits = self(tokens, tgt_key_pad_mask=att_pad_masks)
        loss = self._get_loss(logits, targets)
        self.log(
            "validation/loss",
            loss.item(),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
        # should write callback to log predictions
        # if batch_idx == 0;
        # would be nice for finetuning;

    def on_save_checkpoint(self, checkpoint):
        if self.do_lora:
            lora_module_list = lora_utils.extract_lora_layers_qv(
                my_gpt=self.model, do_ffn=self.hparams["do_ffn"]
            )
            checkpoint["state_dict"] = {
                "lora_module_list": lora_module_list.state_dict()
            }
