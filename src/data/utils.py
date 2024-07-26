import json
from pathlib import Path

import requests
import torch
from torch.utils.data import Dataset


class GPT2Dataset(Dataset):
    def __init__(self, idxs, context_len, stride):
        super().__init__()
        self.input_ids = []
        self.output_ids = []
        # 0, 1, 2, 3, 4, 5
        for i in range(0, len(idxs) - context_len, stride):
            self.input_ids.append(torch.LongTensor(idxs[i : i + context_len]))
            self.output_ids.append(
                torch.LongTensor(idxs[i + 1 : i + context_len + 1])
            )
            assert (
                len(self.input_ids[-1])
                == len(self.output_ids[-1])
                == context_len
            )
        # as if 1d conv on len(idxs) - 1 items;
        assert (
            len(self.input_ids) == (len(idxs) - 1 - context_len) // stride + 1
        )

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]


def download_file(url, filepath: Path):
    the_dir = filepath.parent
    the_dir.mkdir(parents=True, exist_ok=True)
    if filepath.exists():
        print(f"\nFILE ALREADY EXISTS: {filepath}")
        return filepath
    response = requests.get(url)
    if response.status_code == 200:
        with open(filepath, "wb") as file:
            file.write(response.content)
        print(f"FILE DOWNLOADED SUCCESSFULLY TO: {filepath}")
        return filepath
    print(
        f"FAILED DOWNLOAD OF {filepath}! STATUS CODE: {response.status_code}"
    )
    return filepath


def save_to_json(obj, filepath: Path, **kwargs):
    parent_dir = filepath.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as file:
        json.dump(obj, file, **kwargs)


def load_json(filepath: Path):
    with open(filepath, "r") as file:
        obj = json.load(file)
    return obj


def split_and_save_data(data, train_frac, test_frac, data_file_path: Path):
    train_count = int(len(data) * train_frac)
    if train_frac + test_frac < 1:
        test_count = int(len(data) * test_frac)
        val_count = len(data) - train_count - test_count
    else:
        test_count = len(data) - train_count
        val_count = round(0.2 * train_count)
        train_count -= val_count
    assert train_count + val_count + test_count == len(data)
    train_data = data[:train_count]
    test_data = data[train_count : train_count + test_count]
    val_data = data[-val_count:]
    stem = data_file_path.stem
    parent_dir = data_file_path.parent

    save_to_json(train_data, parent_dir / f"{stem}_train.json")
    save_to_json(test_data, parent_dir / f"{stem}_test.json")
    save_to_json(val_data, parent_dir / f"{stem}_val.json")
