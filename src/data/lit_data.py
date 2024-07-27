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
    def __init__(
        self,
        train_frac=0.85,
        test_frac=0.10,
        small_alpaca=True,
        batch_size=8,
        num_workers=1,
        on_gpu=False,
    ):
        super().__init__()
        self.small_alpaca = small_alpaca
        self.train_frac = train_frac
        self.test_frac = test_frac
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.on_gpu = on_gpu

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
                data=data,
                train_frac=self.train_frac,
                test_frac=self.test_frac,
                data_file_path=self.data_file_path,
            )
            save_strings_and_tokens(self.data_file_path, self.tokenizer)
            print(f"DATA PREP DONE...")

    def setup(self, stage):
        assert stage in ("fit", "test")
        stem = self.data_file_path.stem
        parent_dir = self.data_file_path.parent
        if stage == "fit":
            train_tokens_dict = utils.load_json(
                parent_dir / f"{stem}_alpaca_token_dict_train.json"
            )
            val_tokens_dict = utils.load_json(
                parent_dir / f"{stem}_alpaca_token_dict_val.json"
            )
            self.train_dataset = InstructDataset(train_tokens_dict["tokens"])
            self.val_dataset = InstructDataset(val_tokens_dict["tokens"])
        else:
            test_tokens_dict = utils.load_json(
                parent_dir / f"{stem}_alpaca_token_dict_test.json"
            )
            self.test_dataset = InstructDataset(test_tokens_dict["tokens"])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
            collate_fn=collate_fn,
        )


def collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
):
    # max len + 1 to allow an EOS token to be added;
    batch_max_length = max(len(item) + 1 for item in batch)
    in_tokens, out_tokens = [], []
    att_pad_masks = []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        # pad sequences to max_length
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # truncate if needed;
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        # ignore celoss for pad tokens in target;
        ignore_celoss_mask = targets == pad_token_id
        indices = torch.nonzero(ignore_celoss_mask).squeeze()
        # since pad token is the same as EOS token, check
        # all EOS tokens after first EOS are treated as pad tokens;
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # get attention pad_mask for the inputs;
        pad_mask = inputs == pad_token_id
        indices = torch.nonzero(pad_mask).squeeze()
        # True values are ignored, so set the entry corresponding
        # to the first EOS token to False, so it's not ignored;
        if indices.numel() > 0:
            # don't ignore attention for first <endoftext> token;
            pad_mask[indices[0]] = False

        in_tokens.append(inputs)
        out_tokens.append(targets)
        att_pad_masks.append(pad_mask)

    in_tokens = torch.stack(in_tokens)
    out_tokens = torch.stack(out_tokens)
    att_pad_masks = torch.stack(att_pad_masks)
    return in_tokens, out_tokens, att_pad_masks


class InstructDataset(Dataset):
    def __init__(self, tokens):
        super().__init__()
        self.tokens = tokens

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx]


def save_strings_and_tokens(
    data_file_path: Path, tokenizer: tiktoken.Encoding
):
    stem = data_file_path.stem
    parent_dir = data_file_path.parent
    train_flows = utils.load_json(parent_dir / f"{stem}_train.json")
    test_flows = utils.load_json(parent_dir / f"{stem}_test.json")
    val_flows = utils.load_json(parent_dir / f"{stem}_val.json")

    summary = {
        "train_len": len(train_flows),
        "val_len": len(val_flows),
        "test_len": len(test_flows),
    }
    utils.save_to_json(
        summary, metadata.RAW_DATA_DIR / "metadata.json", indent=2
    )

    # get alpaca-token pair formats;
    train_dict = _get_alpaca_token_dict(train_flows, tokenizer)
    test_dict = _get_alpaca_token_dict(test_flows, tokenizer)
    val_dict = _get_alpaca_token_dict(val_flows, tokenizer)

    # save as json;
    utils.save_to_json(
        train_dict, parent_dir / f"{stem}_alpaca_token_dict_train.json"
    )
    utils.save_to_json(
        test_dict, parent_dir / f"{stem}_alpaca_token_dict_test.json"
    )
    utils.save_to_json(
        val_dict, parent_dir / f"{stem}_alpaca_token_dict_val.json"
    )


def _get_alpaca_token_dict(flows, tokenizer: tiktoken.Encoding):
    out = {"strings": [], "tokens": []}
    for flow in flows:
        alpaca = get_alpaca_format(flow)
        tokens = decoding.text_to_ids([alpaca], tokenizer)[0]
        out["strings"].append(alpaca)
        out["tokens"].append(tokens)
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


def get_alpaca_response(flow: dict):
    """
    Return Alpaca-style response string.

    flow should have an 'output' key.
    """
    return f"\n\n### Response:\n{flow['output']}"


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
    output = get_alpaca_response(flow)
    return instruction + output


if __name__ == "__main__":
    dm = LitInstructions()
    dm.prepare_data()
    dm.setup("fit")
    val_loader = dm.val_dataloader()
    x, y, pad = next(iter(val_loader))
    print(f"{x.shape=}, {y.shape=}, {pad.shape}")
    print(x[0], pad[0])
