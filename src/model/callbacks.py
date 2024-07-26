import os
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

import src.data.utils as dutils
import wandb
from src.data.lit_data import get_alpaca_format, get_alpaca_instruction
from src.metadata import metadata
from src.model import decoding

os.environ["TIKTOKEN_CACHE_DIR"] = str(metadata.SAVED_MODELS_PATH)
import tiktoken

TOKENIZER = tiktoken.get_encoding("gpt2")

STEM = metadata.SMALL_DATA_FILEPATH.stem
VAL_FLOWS = dutils.load_json(metadata.RAW_DATA_DIR / f"{STEM}_val.json")


class LogValPredsCallback(Callback):
    def on_validation_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if batch_idx == 0 and dataloader_idx == 0:
            wandb_logger = trainer.logger
            n = 5
            instructs = [
                get_alpaca_instruction(flow) for flow in VAL_FLOWS[:n]
            ]
            targets = [
                f"### Response:\n{flow['output']}" for flow in VAL_FLOWS[:n]
            ]
            preds = []
            for i in range(n):
                with torch.no_grad():
                    ids = decoding.generate_from_single_input(
                        pl_module.model,
                        ids=torch.LongTensor(
                            decoding.text_to_ids([instructs[i]], TOKENIZER)
                        ),
                        max_new_tokens=256,
                        context_len=metadata.BASE_CONFIG["context_length"],
                        eos_id=50256,
                    )
                    pred = decoding.ids_to_text(
                        ids.tolist(), TOKENIZER, to_bytes=True
                    )[0]
                    preds.append(pred[len(instructs[i]) :].strip())

            # log table;
            columns = ["input", "label", "prediction"]
            data = list(zip(instructs, targets, preds))
            wandb_logger.log_text(key="samples", columns=columns, data=data)
