import re

import pytest
from utils import delete_experiment, get_time_in_4_digits

from GPT import train_gpt


@pytest.mark.training
def test_train_gpt():
    exp_name = f"test_train_{get_time_in_4_digits}"
    results = train_gpt.train_gpt(experiment_name=exp_name)
    train_losses = results.metrics_dataframe.train_loss.tolist()
    assert train_losses[0] > train_losses[-1], "Training loss did not decrease"
    # assert overfit on new batch
    pattern = r"(TorchTrainer_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    match = re.search(pattern, results.path)
    delete_experiment(exp_name, [match.group(1)])
