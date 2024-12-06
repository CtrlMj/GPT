import torch
import torch.nn as nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(self, head_size, context_size, n_embed=32, dropout=0.2) -> None:
        super().__init__()
        self.head_size = head_size
        self.WQ = nn.Linear(n_embed, head_size, bias=False)
        self.WK = nn.Linear(n_embed, head_size, bias=False)
        self.WV = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x input of size (B, T, embedding_size)
        """
        B, T, embedding_size = x.shape
        q = self.WQ(x)  # q of shape (B, T, head_size)
        k = self.WK(x)
        v = self.WV(x)

        scores = q @ k.transpose(-2, -1) * self.head_size**-0.5  # scores of shape (B, T, T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, value=-torch.inf)
        attentions = F.softmax(scores, dim=-1)
        attentions = self.dropout(attentions)
        output = attentions @ v  # output of size (B, T, head_size)

        return output


class FeedFroward(nn.Module):
    def __init__(self, n_embed=32, dropout=0.2):
        super().__init__()
        self.feedforward = nn.Sequential(nn.Linear(n_embed, n_embed * 4), nn.ReLU(), nn.Linear(n_embed * 4, n_embed), nn.Dropout(dropout))

    def forward(self, x):
        return self.feedforward(x)


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, context_size, n_embed=32) -> None:
        super().__init__()
        head_size = n_embed // n_heads
        self.MultiHeadAttention = nn.ModuleList([AttentionHead(head_size, context_size) for _ in range(n_heads)])
        self.project = nn.Linear(n_embed, n_embed)
        self.layer_norm1 = nn.LayerNorm(n_embed)
        self.feedForward = FeedFroward()
        self.layer_norm2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        """
        x is of shape (B, T, n_embed)
        """
        x = self.layer_norm1(x)
        attentioned = torch.concat([attention(x) for attention in self.MultiHeadAttention], dim=-1)
        x = x + self.project(attentioned)
        x = self.layer_norm2(x)
        x = x + self.feedForward(x)
        return x


class GPT(nn.Module):
    def __init__(self, n_heads, n_blocks, context_size, vocab_size, n_embed=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed)
        self.pos_embed = nn.Embedding(context_size, n_embed)
        self.decoders = nn.Sequential(*[DecoderBlock(n_heads, context_size) for _ in range(n_blocks)], nn.LayerNorm(n_embed))
        self.context_size = context_size
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets=None):
        """
        x: shape (B, T)
        targets: (B, T)
        """
        B, T = x.shape
        embeddings = self.embedding(x)
        pos_embeds = self.pos_embed(
            torch.arange(
                T,
            )
        )  # T * n_embed
        x = embeddings + pos_embeds
        x = self.decoders(x)
        logits = self.lm_head(x)
        B, T, C = logits.shape  # logits shape (B, T, C)

        if targets is not None:
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
        else:
            return logits, None

    @torch.inference_mode()
    def generate(self, x, max_size=100):
        """
        x of shape (B, T)
        """
        self.eval()
        for step in range(max_size):
            logits, loss = self(x[:, -self.context_size:])
            new_token_logits = logits[:, -1, :]
            probs = F.softmax(new_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            x = torch.concat((x, next_tokens), dim=1)
        return x
