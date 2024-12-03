import tempfile
import os
import typer 

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from ray.data.preprocessor import Preprocessor
from ray.train import CheckpointConfig, DatasetConfig, RunConfig, ScalingConfig
import ray.train as raytrain
from ray.train import Checkpoint, session
from ray.train.torch import TorchCheckpoint, TorchTrainer

from train_gpt import GPT
from typing_extensions import Annotated
from utils import get_batch, read_data

# hyper params
torch.manual_seed(1337)




app = typer.Typer()

def train_step(model, train_data, context_size, batch_size, optimizer, n_steps, device):
    model.train()
    total_loss = 0
    for step in range(n_steps):
        xb, yb = get_batch(context_size, batch_size, split=train_data)
        xb.to(device)
        yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += (loss.detach().item() - total_loss) / (step + 1)
    return total_loss


@torch.no_grad()
def eval_step(model, val_data, context_size, batch_size, n_steps, device):
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
def train_loop_per_worker(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # change position embedding's device if cuda is available
    # Hyperparameters
    dropout = config["dropout"]
    lr = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    n_train_steps = config["n_train_steps"]
    n_eval_steps = config["n_eval_steps"]
    n_heads = config["n_heads"]
    n_blocks = config["n_blocks"]
    context_size = config["context_size"]

    # Get datasets
    torch.manual_seed(1337)  # set seed
    train_data, val_data, encode, decode, vocab_size = read_data()
   
    # Model 
    gpt = GPT(n_heads, n_blocks, vocab_size)
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
        raytrain.report({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, 
                        checkpoint=checkpoint)



@app.command()
def train_gpt(
    dropout: Annotated[float, typer.Option(help="Dropout porbability")] = 0.2,
    n_heads: Annotated[int, typer.Option(help="number of attention heads")] = 4,
    n_blocks: Annotated[int, typer.Option(help="number of decoder blocks")] = 3,
    n_embed: Annotated[int, typer.Option(help="token embedding dimentionality")] = 32,
    eval_iters: Annotated[int, typer.Option(help="Number of iterations for evaluation")] = 200,
    lr: Annotated[float, typer.Option(help="Optimization learning rate")] = 3e-4,
    context_size: Annotated[int, typer.Option(help="Context window size")] = 8,
    batch_size: Annotated[int, typer.Option(help="batch size")] = 64,
    num_epochs: Annotated[int, typer.Option(help="number of epochs")] = 3,
    n_train_steps: Annotated[int, typer.Option(help="number of training steps per epoch")] = 5000,
    n_eval_steps: Annotated[int, typer.Option(help="number of of eval steps per epoch")] = 100
):
    train_config = {
        'dropout': dropout,
        'lr': lr,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'n_train_steps': n_train_steps,
        'n_eval_steps': n_eval_steps,
        'n_heads': n_heads,
        'n_blocks': n_blocks,
        'context_size': context_size,
    }
    scaling_config = ScalingConfig(
        num_workers=6,
        )
    checkpoint_config = CheckpointConfig(num_to_keep=1, 
                                         checkpoint_score_attribute="val_loss", 
                                         checkpoint_score_order="min")
    run_config = RunConfig(name="gpt", 
                           checkpoint_config=checkpoint_config, 
                           storage_path=os.path.abspath("./ray_results"))
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    results = trainer.fit()

    return results.best_checkpoints[0].path
    