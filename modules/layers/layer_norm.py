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
        std = x.std(dim=-1, keepdim=True)
        norm_x = (x-mean) / (std + self.eps)

        return norm_x * self.scale + self.shift