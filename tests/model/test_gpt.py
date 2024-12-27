import pytest
import torch

from gpt import AttentionHead


@pytest.mark.parametrize("input, target", [(torch.rand(size=[4, 8]), torch.randn(size=[4, 8])), (torch.rand(size=[2, 8]), None)])
def test_gpt_output(input, target, gpt):
    logits, loss = gpt(input, target)
    assert logits.size() == input.size()


def test_attention_head():
    attenhead = AttentionHead(head_size=4, context_size=8, n_embed=32, dropout=0.2)
    assert attenhead.tril.size() == torch.Size([8, 8])
    attentioned_output = attenhead(torch.rand(size=(2, 8, 32)))
    assert attentioned_output.size() == torch.Size([2, 8, 4])
