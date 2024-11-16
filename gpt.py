import torch
import torch.nn as nn
from torch.nn import functional as F

# hyper params
torch.manual_seed(1337)
context_size = 8
batch_size = 32
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32


with open("tinyshakespeare.txt", 'r') as f:
    all_text = f.read()

vocab = sorted(list(set(all_text)))
vocab_size = len(vocab)
ctoi = {c: i for i, c in enumerate(vocab)}
itoc = {i: c for i, c in enumerate(vocab)}
encode = lambda s: [ctoi[x] for x in s]
decode = lambda l: ''.join([itoc[i] for i in l])


all_data = torch.tensor(encode(all_text), dtype=torch.long)
n = int(all_data.shape[0]*0.9)
train_data = all_data[:n]
val_data = all_data[n:]


def get_batch(split='train'):
    data = train_data if split == 'train' else val_data
    idx = torch.randint(high=len(data)-context_size, size=(batch_size,))
    x = torch.stack([data[i:i+context_size] for i in idx])
    y = torch.stack([data[i+1:i+context_size+1] for i in idx])
    return x, y


class AttentionHead(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__()
        self.head_size = head_size
        self.WQ = nn.Linear(n_embed, head_size, bias=False)
        self.WK = nn.Linear(n_embed, head_size, bias=False)
        self.WV = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
    
    def forward(self, x):
        """
        x input of size (B, T, embedding_size)
        """
        B, T, embedding_size = x.shape
        q = self.WQ(x)  # q of shape (B, T, head_size)
        k = self.WK(x)
        v = self.WV(x)

        scores = q@k.transpose(-2, -1) * self.head_size**-0.5  # scores of shape (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T]==0, value=-torch.inf)
        attentions = F.softmax(scores, dim=-1)
        output = attentions@v  # output of size (B, T, head_size)

        return output


class FeedFroward(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(n_embed, n_embed)
        self.l2 = nn.Linear(n_embed, n_embed)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, n_heads) -> None:
        super().__init__()
        head_size = n_embed // n_heads
        self.MultiHeadAttention = nn.ModuleList([AttentionHead(head_size) for _ in range(n_heads)])
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.feedForward = FeedFroward()
        self.layer_norm2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        """
        x is of shape (B, T, n_embed)
        """
        attentioned = torch.concat([attention(x) for attention in self.MultiHeadAttention], dim=-1)
        x = self.layer_norm1(x + attentioned)
        forwarded = self.feedForward(x)
        x = self.layer_norm2(x + forwarded)
        return x


class GPT(nn.Module):
    def __init__(self, n_heads, n_blocks):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(context_size, n_embed)
        self.decoders = nn.ModuleList([DecoderBlock(n_heads) for _ in range(n_blocks)])
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        """
        x: shape (B, T)
        targets: (B, T)
        """
        B, T = x.shape
        embeddings = self.embedding(x) 
        pos_embeds = self.pos_embed(torch.arange(T, device=device))  # T * n_embed
        x = embeddings + pos_embeds
        for decoder in self.decoders:
            x = decoder(x)
        logits = self.lm_head(x)
        B, T, C = logits.shape  # logits shape (B, T, C)
        
        if targets is not None:
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None

    def generate(self, x, max_size=100):
        """
        x of shape (B, T)
        """
        self.eval()
        for step in range(max_size):
            logits, loss = self(x[:, -context_size:])
            new_token_logits = logits[:, -1, :]
            probs = F.softmax(new_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)  
            x = torch.concat((x, next_tokens), dim=1)
        return x
    

if __name__ == "__main__":
    m = GPT(n_heads=4, n_blocks=3).to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    m.train()
    for epoch in range(3000):
        xb, yb = get_batch(split='train')
        xb.to(device)
        yb.to(device)
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        print(loss)
        
    generation = m.generate(torch.zeros((1, 1), dtype=torch.long), max_size=200)
    print(decode(generation[0].tolist()))