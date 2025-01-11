import os
import shutil
from datetime import datetime
from typing import List

from GPT.config import SHARED_STORAGE, mlflow


def get_time_in_4_digits():
    """
    Returns the current time in HHMM format, with leading zeros if necessary.

    Returns:
        str: A string representation of the time, e.g., "0635" for 6:35.
    """
    now = datetime.now()
    return f"{now.hour:02d}{now.minute:02d}"


def delete_experiment(experiment_name: str, torchtrainer_name: List[str]):
    mlflowclient = mlflow.tracking.MlflowClient()
    # delete experiment name
    experiment_id = mlflowclient.get_experiment_by_name(experiment_name).experiment_id
    mlflowclient.delete_experiment(experiment_id)
    # delete Ray's TorchTrainer path
    for trainer_name in torchtrainer_name:
        shutil.rmtree(f"{SHARED_STORAGE}/{trainer_name}")
    # delete custom results dictionary
    os.remove(f"{SHARED_STORAGE}/results/{experiment_name}.json")
