import os
from typing import Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback

from src.data.lit_data import get_alpaca_instruction, get_alpaca_response
from src.metadata import metadata
from src.model import decoding

os.environ["TIKTOKEN_CACHE_DIR"] = str(metadata.SAVED_MODELS_PATH)
import tiktoken

TOKENIZER = tiktoken.get_encoding("gpt2")

VAL_FLOWS = [
    {
        "instruction": "Convert the active sentence to passive: 'The chef cooks the meal every day.'",
        "input": "",
        "output": "The meal is cooked by the chef every day.",
    },
    {
        "instruction": "Classify an input string as either a noun or a verb.",
        "input": "Dance",
        "output": "'Dance' can be classified as a verb.",
    },
    {
        "instruction": "Rewrite the sentence using a metaphor.",
        "input": "The book is very interesting.",
        "output": "The book is a page-turner.",
    },
]


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
            n = 3
            instructs = [
                get_alpaca_instruction(flow) for flow in VAL_FLOWS[:n]
            ]
            targets = [
                get_alpaca_response(flow).strip() for flow in VAL_FLOWS[:n]
            ]
            preds = []
            for i in range(n):
                with torch.no_grad():
                    # on a cpu with gpt2-medium one generation of
                    # 35 new tokens costs about 11 secs.
                    ids = decoding.generate_from_single_input(
                        pl_module.model,
                        ids=torch.LongTensor(
                            decoding.text_to_ids([instructs[i]], TOKENIZER)
                        ),
                        temperature=0,  # greedy decoding;
                        max_new_tokens=35,
                        context_len=metadata.BASE_CONFIG["context_length"],
                        eos_id=50256,
                    )
                    pred = decoding.ids_to_text(
                        ids.tolist(), TOKENIZER, to_bytes=True
                    )[0]
                    preds.append(pred[len(instructs[i]) :].strip())

            # log table;
            columns = ["input", "label", "prediction", "epoch"]
            ep = (
                -1
                if pl_module.current_epoch is None
                else pl_module.current_epoch
            )
            epochs = [ep] * len(targets)
            data = list(zip(instructs, targets, preds, epochs))
            wandb_logger.log_text(key="samples", columns=columns, data=data)
