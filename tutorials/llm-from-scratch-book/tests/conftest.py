import torch
from pytest import fixture


@fixture(scope="module")
def batch_3_5_7():
    return torch.randn((3, 5, 7))
