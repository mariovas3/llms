from pytest import fixture
import torch


@fixture(scope='module')
def dummy_embedded_batch():
    return torch.randn((3, 5, 7))