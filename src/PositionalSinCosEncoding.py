import math
import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, t=10000):
        super().__init__()
        self.t = t

    def forward(self, x):
        B, N, C = x.shape
        assert C % 2 == 0, 'dim must be divided 2'

        pos_embed = torch.zeros(N, C, dtype=torch.float32)

        N_num = torch.arange(N, dtype=torch.float32)

        o = torch.arange(C//2, dtype=torch.float32)
        o /= C/2.
        o = 1. / (self.t**o)

        out = N_num[:, None] @ o[None, :]

        sin_embed = torch.sin(out)
        cos_embed = torch.cos(out)

        pos_embed[:, 0::2] = sin_embed
        pos_embed[:, 1::2] = cos_embed

        pos_embed = pos_embed.unsqueeze(0).repeat(B, 1, 1)
        return pos_embed