import os
import tempfile
from typing import Dict

import ray
import ray.train as raytrain
import torch
import typer
from config import MLFLOW_TRACKING_URI, SHARED_STORAGE
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.train import Checkpoint, CheckpointConfig, RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer
from torch.nn.parallel import DistributedDataParallel
from typing_extensions import Annotated
from utils import get_batch, read_data, save_dict

from gpt import GPT

# hyper params
torch.manual_seed(1337)


app = typer.Typer()


def train_step(
    model: torch.nn.Module, train_data: torch.Tensor, context_size: int, batch_size: int, optimizer: torch.optim, n_steps: int, device: str
) -> float:
    """training step

    Args:
        model (torch.nn.Module): model to train
        train_data (torch.Tensor): training data
        context_size (int): context size
        batch_size (int): batch size
        optimizer (torch.optim): optimizer to use for training
        n_steps (int): number of training steps
        device (str): device to train the model on

    Returns:
        float: _description_
    """

    model.train()
    total_loss = 0
    for step in range(n_steps):
        xb, yb = get_batch(context_size, batch_size, split=train_data)
        xb = xb.to(torch.device(device))
        yb = yb.to(torch.device(device))
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += (loss.detach().item() - total_loss) / (step + 1)
    return total_loss


@torch.no_grad()
def eval_step(model: torch.nn.Module, val_data: torch.Tensor, context_size: int, batch_size: int, n_steps: int, device: str) -> float:
    """evaluation step

    Args:
        model (torch.nn.Module): model to run evaluation on
        val_data (torch.Tensor): validatin data
        context_size (int): size of context
        batch_size (int): batch size
        n_steps (int): int
        device (torch.DeviceObjType): device to run evaulation on

    Returns:
        float: total loss of the evaluation step
    """

    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for step in range(n_steps):
            xb, yb = get_batch(context_size, batch_size, split=val_data)
            xb.to(device)
            yb.to(device)
            logits, loss = model(xb, yb)
            total_loss += (loss.item() - total_loss) / (step + 1)
    return total_loss


# Training loop
def train_loop_per_worker(config: Dict) -> None:
    """Training loop per ray cluster worker

    Args:
        config (Dict): configuration dictionary for training.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"  # change position embedding's device if cuda is available
    # Hyperparameters
    lr = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    n_train_steps = config["n_train_steps"]
    n_eval_steps = config["n_eval_steps"]
    n_heads = config["n_heads"]
    n_blocks = config["n_blocks"]
    context_size = config["context_size"]
    n_embed = config["n_embed"]

    # Get datasets
    torch.manual_seed(1337)  # set seed
    train_data, val_data, encode, decode, vocab_size = read_data()

    # Model
    gpt = GPT(n_heads, n_blocks, context_size, vocab_size, n_embed)
    gpt.to(torch.device(device))
    gpt = raytrain.torch.prepare_model(gpt)

    # Training components
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=lr)

    # Training
    batch_size_per_worker = batch_size // raytrain.get_context().get_world_size()
    for epoch in range(num_epochs):
        # Step
        train_loss = train_step(gpt, train_data, context_size, batch_size_per_worker, optimizer, n_train_steps, device)
        val_loss = eval_step(gpt, val_data, context_size, batch_size_per_worker, n_eval_steps, device)

        # Checkpoint
        base_model = gpt.module if isinstance(gpt, DistributedDataParallel) else gpt
        checkpoint_dir = tempfile.mkdtemp()
        torch.save(
            {"model_state_dict": base_model.state_dict()},
            os.path.join(checkpoint_dir, "model.pt"),
        )
        checkpoint = Checkpoint.from_directory(checkpoint_dir)

        # Report metrics and checkpoint.
        raytrain.report({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, checkpoint=checkpoint)


@app.command()
def train_gpt(
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
    """Main training function

    Args:
        experiment_name (Annotated[str, typer.Option, optional): experiment name. Defaults to "Name for mlflow experiment")].
        dropout (Annotated[float, typer.Option, optional): dropout probability. Defaults to "Dropout porbability")]=0.2.
        n_heads (Annotated[int, typer.Option, optional): number of attention heads. Defaults to "number of attention heads")]=4.
        n_blocks (Annotated[int, typer.Option, optional): number of decoder blocks. Defaults to "number of decoder blocks")]=3.
        n_embed (Annotated[int, typer.Option, optional): embedding dimension. Defaults to "token embedding dimentionality")]=32.
        lr (Annotated[float, typer.Option, optional): learning rate. Defaults to "Optimization learning rate")]=3e-4.
        context_size (Annotated[int, typer.Option, optional): context size. Defaults to "Context window size")]=8.
        batch_size (Annotated[int, typer.Option, optional): batch size. Defaults to "batch size")]=64.
        num_epochs (Annotated[int, typer.Option, optional): number of epochs. Defaults to "number of epochs")]=3.
        n_train_steps (Annotated[int, typer.Option, optional): number of training steps. Defaults to "number of training steps per epoch")]=5000.
        n_eval_steps (Annotated[int, typer.Option, optional): number of evaluation step. Defaults to "number of of eval steps per epoch")]=100.
        n_workers (Annotated[int, typer.Option, optional): number of cluster workers allocated for training. Defaults to "number of worker nodes for training")]=1.

    Returns:
        str: path to the checkpoint
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
    scaling_config = ScalingConfig(
        num_workers=n_workers,
    )
    checkpoint_config = CheckpointConfig(num_to_keep=1, checkpoint_score_attribute="val_loss", checkpoint_score_order="min")

    mlflowcallback = MLflowLoggerCallback(tracking_uri=MLFLOW_TRACKING_URI, experiment_name=experiment_name, save_artifact=True)
    run_config = RunConfig(checkpoint_config=checkpoint_config, storage_path=str(SHARED_STORAGE.absolute()), callbacks=[mlflowcallback])

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    results = trainer.fit()
    results_d = {
        "best_checkpoint_dir": results.best_checkpoints[0].checkpoint.path,
        "experiment_name": experiment_name,
    }
    save_dict(results_d, os.path.abspath(f"{str(SHARED_STORAGE.absolute())}/results/{experiment_name}"))

    return results.best_checkpoints[0].checkpoint.path


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()
