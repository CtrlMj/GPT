import json
import os
from typing import Dict

import torch

from gpt import GPT


def read_data():
    """reads the data and it's properties

    Returns:
        _type_: returns the data, and properties such as vocab size and decoding functions.
    """
    with open("/Users/majid/Projects/nlp/GPT/tinyshakespeare.txt", "r", encoding="utf-8") as f:
        allshakespeare = f.read()

    vocab = sorted(list(set(allshakespeare)))
    ctoi = {c: i for i, c in enumerate(vocab)}
    itoc = {i: c for i, c in enumerate(vocab)}
    encode = lambda s: [ctoi[x] for x in s]
    decode = lambda lst: "".join([itoc[x] for x in lst])
    n = int(len(allshakespeare) * 0.9)
    train_data = torch.tensor(encode(allshakespeare[:n]), dtype=torch.long)
    val_data = torch.tensor(encode(allshakespeare[n:]), dtype=torch.long)
    return train_data, val_data, encode, decode, len(vocab)


train_data, val_data, encode, decode, vocab_size = read_data()


def get_batch(context_size: int, batch_size: int, split: torch.Tensor):
    """batch data

    Args:
        context_size (int): context size of the language model
        batch_size (int): size of the batch
        split (torch.Tensor): tensor data set to draw batch from

    Returns:
        (torch.Tensor, torch.Tensor): a batch of input tensor and predictions
    """
    idx = torch.randint(high=len(split) - context_size, size=(batch_size,))
    x = torch.stack([split[i : i + context_size] for i in idx])
    y = torch.stack([split[i + 1 : i + context_size + 1] for i in idx])
    return x, y


def save_dict(dic: Dict, path: str, filename: str) -> None:
    """saves a dictionary
    note: filename is a separate arg from path
    otherwise after mkdirs creates the path
    python would complain when writing into that path:
    IsADirectoryError: [Errno 21] Is a directory:


    Args:
        dic (Dict): dictionary to save
        path (str): path to save the dictionary
        filename (str): name of the file in that path
    """
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"{path}/{filename}", "w", encoding="utf-8") as f:
        json.dump(dic, indent=2, fp=f)


def load_model_from_checkpoint(checkpoint_path: str, config: dict) -> torch.nn.Module:
    """loads model from checkpoint path

    Args:
        checkpoint_path (str): path to the checkpoint
        config (dict): dictionary with model properties

    Returns:
        torch.nn.Module: GPT model to return
    """
    model = GPT(config["n_heads"], config["n_blocks"], config["context_size"], vocab_size, config["n_embed"])
    model_state = torch.load(checkpoint_path + "/model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(model_state["model_state_dict"])
    return model
