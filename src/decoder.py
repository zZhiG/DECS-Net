import torch
import torch.nn as nn
from transformer import Block1
from dsc import IDSC


class decoder(nn.Module):
    def __init__(self, num_class, dims):
        super().__init__()

        self.p = nn.PixelShuffle(2)


        self.c1 = IDSC(dims[3]//4 + dims[2], dims[2])
        self.b1 = Block1(dims[2], 8, 4, 0.2)


        self.c2 = IDSC(dims[2]//4 + dims[1], dims[1])
        self.b2 = Block1(dims[1], 8, 4, 0.3)


        self.c3 = IDSC(dims[1]//4 + dims[0], dims[0])
        self.b3 = Block1(dims[0], 8, 8, 0.4)


        self.head = nn.Sequential(nn.PixelShuffle(4),
                                  nn.Conv2d(dims[0]//16, num_class, 1))

    def forward(self, x1, x2, x3, x4):

        x4 = self.p(x4)
        x3 = torch.cat([x3, x4], dim=1)
        x3 = self.c1(x3)
        b, c, h, w = x3.shape
        x3 = self.b1(x3.reshape(b, c, -1).permute(0, 2, 1))
        x3 = x3.permute(0, 2, 1).reshape(b, c, h, w)

        x3 = self.p(x3)
        x2 = torch.cat([x2, x3], dim=1)
        x2 = self.c2(x2)
        b, c, h, w = x2.shape
        x2 = self.b2(x2.reshape(b, c, -1).permute(0, 2, 1))
        x2 = x2.permute(0, 2, 1).reshape(b, c, h, w)

        x2 = self.p(x2)
        x1 = torch.cat([x1, x2], dim=1)
        x1 = self.c3(x1)
        b, c, h, w = x1.shape
        x1 = self.b3(x1.reshape(b, c, -1).permute(0, 2, 1))
        x1 = x1.permute(0, 2, 1).reshape(b, c, h, w)

        out = self.head(x1)
        return out