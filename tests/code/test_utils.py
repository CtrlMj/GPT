import pytest
import torch

from utils import get_batch, read_data


@pytest.mark.unit
def test_read_data():
    _, _, encode, decode, vocab_size = read_data()
    assert vocab_size == 65, "vocabulary size is not the expected 65"
    assert decode([39, 40, 41]) == "abc", "decoder not working properly"
    assert encode("abc") == [39, 40, 41], "encoder not working properly"


@pytest.mark.unit
def test_get_batch():
    split = torch.tensor([23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123])
    context_size = 4
    batch_size = 5
    x, y = get_batch(context_size, batch_size, split)
    assert x.size() == torch.Size([batch_size, context_size])
    assert y.size() == torch.Size([batch_size, context_size])
