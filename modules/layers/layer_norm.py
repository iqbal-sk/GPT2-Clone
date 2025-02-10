import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, embedding_size):
        super(LayerNorm, self).__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(embedding_size))
        self.shift = nn.Parameter(torch.zeros(embedding_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        return self.scale * norm_x + self.shift