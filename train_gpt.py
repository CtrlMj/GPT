import torch
import torch.nn as nn
from torch.nn import functional as F

from ray.data.preprocessor import Preprocessor
from ray.air.config import CheckpointConfig, DatasetConfig, RunConfig, ScalingConfig
import ray.train as raytrain
from ray.train import Checkpoint, session
from ray.train.torch import TorchCheckpoint, TorchTrainer

# hyper params
torch.manual_seed(1337)
context_size = 8
batch_size = 64
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # change position embedding's device if cuda is available
eval_iters = 200
n_embed = 32
n_head = 4
dropout = 0.2

def train_step(model, train_data, batch_size, optimizer, n_steps):
    model.train()
    total_loss = 0
    for step in range(n_steps):
        xb, yb = get_batch(train_data, split='train')
        xb.to(device)
        yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total_loss += (loss.detach().item() - total_loss) / (step + 1)
    return total_loss


@torch.no_grad()
def eval_step(model, val_data, batch_size, n_steps):
    model.eval()
    total_loss = 0
    with torch.inference_mode():
        for step in range(n_steps):
            xb, yb = get_batch(val_data, split='eval')
            xb.to(device)
            yb.to(device)
            logits, loss = model(xb, yb)
            total_loss += (loss.item() - total_loss) / (step + 1)
    return total_loss


# Training loop
def train_loop_per_worker(config):
    # Hyperparameters
    dropout_p = config["dropout_p"]
    lr = config["lr"]
    num_epochs = config["num_epochs"]
    batch_size = config["batch_size"]
    n_train_steps = config["n_train_steps"]
    n_eval_steps = config["n_eval_steps"]
    n_heads = config["n_heads"]
    n_blocks = config["n_blocks"]

    # Get datasets
    torch.manual_seed(1337) #set seed
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
        train_loss = train_step(gpt, train_data, batch_size_per_worker, optimizer, n_train_steps)
        val_loss = eval_step(gpt, val_data, batch_size_per_worker, n_eval_steps)

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


def train_gpt():
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
    