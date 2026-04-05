import pytest
import torch


def test_torch_version():
    assert torch.__version__ is not None
