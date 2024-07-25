import torch
from torch.utils.data import Dataset


class GPT2Dataset(Dataset):
    def __init__(self, idxs, context_len, stride):
        super().__init__()
        self.input_ids = []
        self.output_ids = []
        # 0, 1, 2, 3, 4, 5
        for i in range(0, len(idxs) - context_len, stride):
            self.input_ids.append(torch.LongTensor(idxs[i:i + context_len]))
            self.output_ids.append(torch.LongTensor(idxs[i+1:i+context_len+1]))
            assert len(self.input_ids[-1]) == len(self.output_ids[-1]) == context_len
        # as if 1d conv on len(idxs) - 1 items;
        assert len(self.input_ids) == (len(idxs) - 1 - context_len) // stride + 1

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.output_ids[idx]

