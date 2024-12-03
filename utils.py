import os
import random
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



def get_batch(context_size, batch_size, split):
    idx = torch.randint(high=len(split)-context_size, size=(batch_size,))
    x = torch.stack([split[i:i+context_size] for i in idx])
    y = torch.stack([split[i+1:i+context_size+1] for i in idx])
    return x, y


