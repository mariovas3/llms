import torch
from torch import nn


class PosEmbed(nn.Module):
    def __init__(self, num_embeds, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeds, embed_dim)
    
    def forward(self, x):
        return x + self.embed(torch.tensor(range(x.size(-2))))


def get_torch_mask(size):
    """
    Returns square boolean mask.

    True values will be ignored in attention.

    The lower triangle is False, the remaining entries are True.
    """
    return torch.triu(torch.ones((size, size)), diagonal=1).bool()
