from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.layerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.layerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self, ln_1(x))
        x = x + self.mlp(self, ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer   : int = 6
    n_head    : int = 6
    n_embd    : int = 384

class GPT(nn.Module):

    def __init__(self, config):
        super.__init__()
        self.config = config

    self.transformer = nn.ModuleDict(dict(
        wte  = nn.Embedding(config.vocab_size, config.n_embd),
        wpe  = nn.Embedding(config.block_size, config.n_embd)
        h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        ln_f = nn.LayerNorm(config.n_embd),
    ))

    self.lm_head = nn.Linear(config.n_head, config.vocab_size, bias=False)