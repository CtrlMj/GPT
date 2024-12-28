import pytest
from utils import delete_experiment

from GPT import train_gpt


@pytest.mark.training
def test_train_gpt():
    results = train_gpt.train_gpt(experiment_name="test")
    delete_experiment("test", results.path)
    train_losses = results.metrics_dataframe.train_loss.tolist()
    assert train_losses[0] > train_losses[-1], "Training loss did not decrease"
    # assert overfit on new batch
