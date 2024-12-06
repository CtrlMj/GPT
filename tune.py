import os

import ray
import typer
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from ray.tune import TuneConfig, Tuner
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.hyperopt import HyperOptSearch
from typing_extensions import Annotated

from config import MLFLOW_TRACKING_URI, SHARED_STORAGE
from train_gpt import train_loop_per_worker
from utils import save_dict

app = typer.Typer()


@app.command()
def tune_gpt(
    experiment_name: Annotated[str, typer.Option(help="Name for mlflow experiment")],
    dropout: Annotated[float, typer.Option(help="Dropout porbability")] = 0.2,
    n_heads: Annotated[int, typer.Option(help="number of attention heads")] = 4,
    n_blocks: Annotated[int, typer.Option(help="number of decoder blocks")] = 3,
    n_embed: Annotated[int, typer.Option(help="token embedding dimentionality")] = 32,
    lr: Annotated[float, typer.Option(help="Optimization learning rate")] = 3e-4,
    context_size: Annotated[int, typer.Option(help="Context window size")] = 8,
    batch_size: Annotated[int, typer.Option(help="batch size")] = 64,
    num_epochs: Annotated[int, typer.Option(help="number of epochs")] = 3,
    n_train_steps: Annotated[int, typer.Option(help="number of training steps per epoch")] = 5000,
    n_eval_steps: Annotated[int, typer.Option(help="number of of eval steps per epoch")] = 100,
    n_workers: Annotated[int, typer.Option(help="number of worker nodes for training")] = 1,
) -> str:
    """tuning function

    Args:
        experiment_name (Annotated[str, typer.Option, optional): name of experiment. Defaults to "Name for mlflow experiment")].
        dropout (Annotated[float, typer.Option, optional): droput probability. Defaults to "Dropout porbability")]=0.2.
        n_heads (Annotated[int, typer.Option, optional): number of attention heads. Defaults to "number of attention heads")]=4.
        n_blocks (Annotated[int, typer.Option, optional): number of decoder blocks. Defaults to "number of decoder blocks")]=3.
        n_embed (Annotated[int, typer.Option, optional): number of embedding dimension. Defaults to "token embedding dimentionality")]=32.
        lr (Annotated[float, typer.Option, optional): learning rate. Defaults to "Optimization learning rate")]=3e-4.
        context_size (Annotated[int, typer.Option, optional): size of context. Defaults to "Context window size")]=8.
        batch_size (Annotated[int, typer.Option, optional): batch size. Defaults to "batch size")]=64.
        num_epochs (Annotated[int, typer.Option, optional): number of epochs for training. Defaults to "number of epochs")]=3.
        n_train_steps (Annotated[int, typer.Option, optional): number of training steps. Defaults to "number of training steps per epoch")]=5000.
        n_eval_steps (Annotated[int, typer.Option, optional): number of evaluation steps. Defaults to "number of of eval steps per epoch")]=100.
        n_workers (Annotated[int, typer.Option, optional): number of cluster workers to allocate for training. Defaults to "number of worker nodes for training")]=1.

    Returns:
        str: checkpoint path
    """
    train_config = {
        "dropout": dropout,
        "lr": lr,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "n_train_steps": n_train_steps,
        "n_eval_steps": n_eval_steps,
        "n_heads": n_heads,
        "n_blocks": n_blocks,
        "n_embed": n_embed,
        "context_size": context_size,
    }

    # training config
    scaling_config = ScalingConfig(
        num_workers=n_workers,
    )

    trainer = TorchTrainer(train_loop_per_worker=train_loop_per_worker, train_loop_config=train_config, scaling_config=scaling_config)

    # run config
    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min")

    mlflowcallback = MLflowLoggerCallback(tracking_uri=MLFLOW_TRACKING_URI, experiment_name=experiment_name, save_artifact=True)

    run_config = RunConfig(callbacks=[mlflowcallback], checkpoint_config=checkpoint_config, storage_path=str(SHARED_STORAGE.absolute()))

    # tune config

    initial_params = [{"train_loop_config": {"dropout_p": 0.2, "lr": 3e-4, "n_train_steps": 2000}}]
    search_alg = HyperOptSearch(points_to_evaluate=initial_params)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    scheduler = AsyncHyperBandScheduler(
        max_t=train_config["num_epochs"],
        grace_period=1,
    )

    tune_config = TuneConfig(
        metric="val_loss",
        mode="min",
        search_alg=search_alg,
        scheduler=scheduler,
        num_samples=2,
    )

    tuner = Tuner(
        trainable=trainer,
        run_config=run_config,
        tune_config=tune_config,
        param_space={
            "train_loop_config": {
                "dropout_p": tune.uniform(0.1, 0.8),
                "lr": tune.loguniform(1e-5, 5e-4),
                "n_train_steps": tune.randint(1000, 4000),
            }
        },
    )

    results = tuner.fit()

    best_trial = results.get_best_result(metric="val_loss", mode="min")
    results_d = {
        "best_tunning_checkpoint_dir": best_trial.checkpoint.path,
        "experiment_name": experiment_name,
        "n_heads": n_heads,
        "n_blocks": n_blocks,
        "n_embed": n_embed,
        "context_size": context_size,
    }
    save_dict(results_d, os.path.abspath(f"./results/{experiment_name}"))

    return best_trial.checkpoint.path


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
