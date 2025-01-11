import re

import pytest
from utils import delete_experiment, get_time_in_4_digits

from GPT.tune import tune_gpt


@pytest.mark.training
def test_tune_gpt():
    exp_name = f"test_tune_{get_time_in_4_digits}"
    results = tune_gpt(experiment_name=exp_name)
    assert results.num_errors == 0, f"Some errors have occurred during tuning:, {results.errors}"
    pattern = r"(TorchTrainer_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    match = re.search(pattern, results[0].path)
    delete_experiment(exp_name, [match.group(1)])
