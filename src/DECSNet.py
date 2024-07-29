import torch
import torch.nn as nn
from encoder import Encoder
from decoder import decoder

class DECSNet(nn.Module):
    def __init__(self, in_channel, out_channel, resolution, patchsz, mode, pretrained, cnn_dim, transform_dim, num_class):
        super().__init__()
        self.encoder = Encoder(in_channel, out_channel, resolution, patchsz, mode, pretrained, cnn_dim, transform_dim)
        self.decoder = decoder(num_class, transform_dim)

    def forward(self, x):
        o1, o2, o3, o4 = self.encoder(x)
        out = self.decoder(o1, o2, o3, o4)

        return out
