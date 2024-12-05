import os
import random
import torch
import json
from gpt import GPT
import torch

def read_data():
    with open('/Users/majid/Projects/nlp/GPT/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
        allshakespeare = f.read()

    vocab = sorted(list(set(allshakespeare)))
    ctoi = {c: i for i, c in enumerate(vocab)}
    itoc = {i: c for i, c in enumerate(vocab)}
    encode = lambda s: [ctoi[x] for x in s]
    decode = lambda l: ''.join([itoc[x] for x in l])
    n = int(len(allshakespeare)*0.9)
    train_data = torch.tensor(encode(allshakespeare[:n]), dtype=torch.long)
    val_data = torch.tensor(encode(allshakespeare[n:]), dtype=torch.long)
    return train_data, val_data, encode, decode, len(vocab)


train_data, val_data, encode, decode, vocab_size = read_data()


def get_batch(context_size, batch_size, split):
    idx = torch.randint(high=len(split)-context_size, size=(batch_size,))
    x = torch.stack([split[i:i+context_size] for i in idx])
    y = torch.stack([split[i+1:i+context_size+1] for i in idx])
    return x, y


def save_dict(dic, path):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dic, indent=2, fp=f)



def load_model_from_checkpoint(checkpoint_path, config):
    model = GPT(config['n_heads'], config['n_blocks'], config['context_size'], vocab_size, config['n_embed'])
    model_state = torch.load(checkpoint_path+"/model.pt", map_location='cpu', weights_only=True)
    model.load_state_dict(model_state['model_state_dict'])
    return model

