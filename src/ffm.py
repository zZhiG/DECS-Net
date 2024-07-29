import torch
import torch.nn as nn
from dsc import DSC, IDSC

class FFM(nn.Module):
    def __init__(self, dim1, dim2, res):
        super().__init__()
        self.trans_c = nn.Conv2d(dim1, dim2, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.li1 = nn.Linear(dim2, dim2)
        self.li2 = nn.Linear(dim2, dim2)

        self.qx = DSC(dim2, dim2)
        self.kx = DSC(dim2, dim2)
        self.vx = DSC(dim2, dim2)
        self.projx = DSC(dim2, dim2)

        self.qy = DSC(dim2, dim2)
        self.ky = DSC(dim2, dim2)
        self.vy = DSC(dim2, dim2)
        self.projy = DSC(dim2, dim2)

        self.concat = nn.Conv2d(dim2*2, dim2, 1)

        self.fusion = nn.Sequential(IDSC(dim2*4, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    DSC(dim2, dim2),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU(),
                                    nn.Conv2d(dim2, dim2, 1),
                                    nn.BatchNorm2d(dim2),
                                    nn.GELU())


    def forward(self, x, y):
        b, c, h, w = x.shape
        B, N, C = y.shape
        H = W = int(N**0.5)

        x = self.trans_c(x)
        y = y.reshape(B, H, W, C).permute(0, 3, 1, 2)

        avg_x = self.avg(x).permute(0, 2, 3, 1)
        avg_y = self.avg(y).permute(0, 2, 3, 1)
        x_weight = self.li1(avg_x)
        y_weight = self.li2(avg_y)
        x = x.permute(0, 2, 3, 1) * x_weight
        y = y.permute(0, 2, 3, 1) * y_weight

        out1 = x * y
        out1 = out1.permute(0, 3, 1, 2)

        x = x.permute(0, 3, 1, 2)
        y = y.permute(0, 3, 1, 2)

        qy = self.qy(y).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)
        kx = self.kx(x).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)
        vx = self.vx(x).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)

        attnx = (qy @ kx.transpose(-2, -1)) * (C**-0.5)
        attnx = attnx.softmax(dim=-1)
        attnx = (attnx @ vx).transpose(2, 3).reshape(B, H//4, w//4, 4, 4, C)
        attnx = attnx.transpose(2, 3).reshape(B,  H, W, C).permute(0, 3, 1, 2)
        attnx = self.projx(attnx)


        qx = self.qx(x).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)
        ky = self.ky(y).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)
        vy = self.vy(y).reshape(B, 8, C//8, H//4, 4, W//4, 4).permute(0, 3, 5, 1, 4, 6, 2).reshape(B, N//16, 8, 16, C//8)

        attny = (qx @ ky.transpose(-2, -1)) * (C**-0.5)
        attny = attny.softmax(dim=-1)
        attny = (attny @ vy).transpose(2, 3).reshape(B, H//4, w//4, 4, 4, C)
        attny = attny.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)
        attny = self.projy(attny)

        out2 = torch.cat([attnx, attny], dim=1)
        out2 = self.concat(out2)

        out = torch.cat([x, y, out1, out2], dim=1)

        out = self.fusion(out)
        return out
