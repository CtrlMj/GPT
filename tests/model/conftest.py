import pytest

from gpt import GPT


@pytest.fixture
def gpt():
    return GPT(n_heads=2, n_blocks=2, context_size=8, vocab_size=65, n_embed=32)
