import os
from pathlib import Path

import tiktoken
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data import utils
from src.metadata import metadata
from src.model import decoding

os.environ["TIKTOKEN_CACHE_DIR"] = str(metadata.SAVED_MODELS_PATH)


class LitInstructions(LightningDataModule):
    def __init__(self, train_frac=0.85, test_frac=0.10, small_alpaca=True):
        super().__init__()
        self.small_alpaca = small_alpaca
        self.train_frac = train_frac
        self.test_frac = test_frac
        self.tokenizer = tiktoken.get_encoding("gpt2")

        if small_alpaca:
            self.data_file_path = metadata.SMALL_DATA_FILEPATH
            self.data_url = metadata.SMALL_DATA_URL
        else:
            raise NotImplementedError("small_alpaca=False is not implemented")

    def prepare_data(self):
        if self.data_file_path.exists():
            print(f"FILE ALREADY EXISTS {self.data_file_path}")
        else:
            print(f"DOWNLOADING ALPACA DATA...")
            utils.download_file(self.data_url, self.data_file_path)
            data = utils.load_json(self.data_file_path)
            print("SPLITTING AND SAVING DATA...")
            utils.split_and_save_data(
                data,
                self.train_frac,
                self.data_file_path,
                test_frac=self.test_frac,
            )
            save_strings_and_tokens(self.data_file_path, self.tokenizer)
            print(f"DATA PREP DONE...")

    def setup(self, split):
        pass


class InstructDataset(Dataset):
    pass


def save_strings_and_tokens(
    data_file_path: Path, tokenizer: tiktoken.Encoding
):
    stem = data_file_path.stem
    parent_dir = data_file_path.parent
    train_flows = utils.load_json(parent_dir / f"stem_train.json")
    test_flows = utils.load_json(parent_dir / f"stem_test.json")
    val_flows = utils.load_json(parent_dir / f"stem_val.json")

    # get alpaca-token pair formats;
    train_formats = _get_alpaca_token_pairs(train_flows, tokenizer)
    test_formats = _get_alpaca_token_pairs(test_flows, tokenizer)
    val_formats = _get_alpaca_token_pairs(val_flows, tokenizer)

    # save as json;
    utils.save_to_json(
        train_formats, parent_dir / f"{stem}_alpaca_token_pairs_train.json"
    )
    utils.save_to_json(
        test_formats, parent_dir / f"{stem}_alpaca_token_pairs_test.json"
    )
    utils.save_to_json(
        val_formats, parent_dir / f"{stem}_alpaca_token_pairs_val.json"
    )


def _get_alpaca_token_pairs(flows, tokenizer: tiktoken.Encoding):
    out = []
    for flow in flows:
        alpaca = get_alpaca_format(flow)
        tokens = decoding.text_to_ids(alpaca, tokenizer)
        out.append([alpaca, tokens])
    return out


def get_alpaca_instruction(flow: dict):
    """
    Gets Alpaca style formatting of flow.

    flow is expected to have an 'instruction' and 'input' keys.

    The Alpaca style is:

    Below is an instruction that describes a task. Write a
    response that appropriately completes the request.

    ### Instruction:
    The instruction.

    ### Input:
    The input - this is an optional field for Alpaca.
    """
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{flow['instruction']}"
    )
    input_text = f"\n\n### Input:\n{flow['input']}" if flow["input"] else ""
    return instruction_text + input_text


def get_alpaca_format(flow: dict):
    """
    Gets Alpaca style formatting of flow.

    flow is expected to have an 'instruction', 'input'
    and 'output' keys.

    The Alpaca style is:

    Below is an instruction that describes a task. Write a
    response that appropriately completes the request.

    ### Instruction:
    The instruction.

    ### Input:
    The input - this is an optional field for Alpaca.

    ### Response:
    The response.
    """
    instruction = get_alpaca_instruction(flow)
    output = f"\n\n### Response:\n{flow['output']}"
    return instruction + output
