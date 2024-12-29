import shutil

from GPT.config import mlflow


def delete_experiment(experiment_name, torchtrainer_name):
    mlflowclient = mlflow.tracking.MlflowClient()
    experiment_id = mlflowclient.get_experiment_by_name(experiment_name).experiment_id
    mlflowclient.delete_experiment(experiment_id)

    path_to_delete = "/".join(torchtrainer_name.split("/")[:-1])
    shutil.rmtree(path_to_delete)
