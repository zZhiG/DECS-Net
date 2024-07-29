import torch
import torch.nn as nn
import torch.nn.functional as F
from dsc import DSC, IDSC
from pytorch_wavelets import DWTForward
from PositionalSinCosEncoding import PositionEmbedding

class PatchEmbedding(nn.Module):
    def __init__(self, dim=3, resolution=512, hidden_dim=96, patchsize=4, norm=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.ps = patchsize
        # self.pos = PositionEmbedding()
        self.embedding = DSC(dim, hidden_dim, patchsize, patchsize, 0)
        self.norm = nn.BatchNorm2d(hidden_dim) if norm == None else norm

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.embedding(x)
        x = self.norm(x)
        x = x.reshape(B, self.hidden_dim, (H//self.ps)*(W//self.ps)).permute(0, 2, 1)
        # x = x + self.pos(x).to(x)
        return x

class PatchMerge(nn.Module):
    def __init__(self, dim, ws=2, norm=None):
        super().__init__()
        self.merge = DSC(dim, dim*2, ws, ws, 0)
        self.norm = nn.BatchNorm2d(dim*2) if norm==None else norm
        # self.pos = PositionEmbedding()

    def forward(self, x):
        B, N, C = x.shape
        h = w = int(N**0.5)
        x = x.permute(0, 2, 1).reshape(B, C, h, w)
        x = self.norm(self.merge(x))
        x = x.reshape(B, C*2, -1).permute(0, 2, 1)
        # x = x + self.pos(x).to(x)

        return x

class HiLo1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim

        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim

        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim

        self.ws = window_size

        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        if self.ws != 1:
            # self.wt = DWTForward(J=1, mode='zero', wave='haar')
            '''
            Because we only need low-frequency components, we can choose to implement by average pooling,
            this will be more convenient to use and understand.
            '''
            self.wt = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.sr_up = nn.PixelShuffle(window_size)
            self.restore = nn.Conv2d(self.dim//(window_size*window_size), self.dim, 1)
        else:
            self.sr = nn.Sequential()
            self.sr_up = nn.Sequential()
            self.restore = nn.Sequential()

        if self.l_heads > 0:
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hi_lofi(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.ws != 1:
            # low_feats, yH = self.wt(x)
            low_feats = self.wt(x) # implementation by average pooling
        else:
            low_feats = self.sr(x)

        low2high = self.sr_up(low_feats)
        high_feats = self.restore(low2high)
        high_feats = high_feats - x

        x = x.permute(0, 2, 3, 1)
        low_feats = low_feats.permute(0, 2, 3, 1)

        if self.l_heads!=0:

            l_q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
            if self.ws > 1:
                l_kv = self.l_kv(low_feats).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
            else:
                l_kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
            l_k, l_v = l_kv[0], l_kv[1]

            l_attn = (l_q @ l_k.transpose(-2, -1)) * self.scale
            l_attn = l_attn.softmax(dim=-1)

            l_x = (l_attn @ l_v).transpose(1, 2).reshape(B, H, W, self.l_dim)
            l_x = self.l_proj(l_x)

        if self.h_heads!=0:
            h_group, w_group = H // self.ws, W // self.ws
            total_groups = h_group * w_group
            high_feats = high_feats.permute(0, 2, 3, 1).reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)
            h_qkv = self.h_qkv(high_feats).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1,
                                                                                                                  4, 2, 5)
            h_q, h_k, h_v = h_qkv[0], h_qkv[1], h_qkv[2]

            h_attn = (h_q @ h_k.transpose(-2, -1)) * self.scale
            h_attn = h_attn.softmax(dim=-1)
            h_attn = (h_attn @ h_v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
            h_x = h_attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

            h_x = self.h_proj(h_x)

        if self.h_heads!=0 and self.l_heads!=0:
            out = torch.cat([l_x, h_x], dim=-1)
            out = out.reshape(B, N, C)

        if self.l_heads==0:
            out = h_x.reshape(B, N, C)
        if self.h_heads==0:
            out = l_x.reshape(B, N, C)

        return out

    def forward(self, x):
        return self.hi_lofi(x)


class FCHiLo1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim / num_heads)
        self.dim = dim
        self.pos = PositionEmbedding()

        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * head_dim

        self.h_heads = num_heads - self.l_heads
        self.h_dim = self.h_heads * head_dim

        self.ws = window_size

        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        if self.ws != 1:
            # self.wt = DWTForward(J=1, mode='zero', wave='haar')
            self.wt = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
        else:
            self.sr = nn.Sequential()

        if self.l_heads > 0:
            self.l_q = DSC(self.dim, self.l_dim)
            self.l_kv = DSC(self.dim, self.l_dim*2)
            self.l_proj = DSC(self.l_dim, self.l_dim)

        if self.h_heads > 0:
            self.h_qkv = DSC(self.dim, self.h_dim*3)
            self.h_proj = DSC(self.h_dim, self.h_dim)

    def hi_lofi(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        if self.ws != 1:
            # low_feats, yH = self.wt(x)
            low_feats = self.wt(x)
        else:
            low_feats = self.sr(x)

        high_feats = F.interpolate(low_feats, size=H, mode='nearest')
        high_feats = high_feats - x

        if self.l_heads!=0:
            l_q = self.l_q(x).permute(0, 2, 3, 1).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
            if self.ws > 1:
                l_kv = self.l_kv(low_feats).permute(0, 2, 3, 1).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
            else:
                l_kv = self.l_kv(x).permute(0, 2, 3, 1).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
            l_k, l_v = l_kv[0], l_kv[1]

            l_attn = (l_q @ l_k.transpose(-2, -1)) * self.scale
            l_attn = l_attn.softmax(dim=-1)

            l_x = (l_attn @ l_v).transpose(1, 2).reshape(B, H, W, self.l_dim).permute(0, 3, 1, 2)
            l_x = self.l_proj(l_x).permute(0, 2, 3, 1)


        if self.h_heads!=0:
            h_group, w_group = H // self.ws, W // self.ws
            total_groups = h_group * w_group
            h_qkv = self.h_qkv(high_feats).permute(0, 2, 3, 1).\
                reshape(B, h_group, self.ws, w_group, self.ws, 3*self.h_dim).\
                transpose(2, 3).reshape(B, total_groups, -1, 3, self.h_heads,
                                        self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
            h_q, h_k, h_v = h_qkv[0], h_qkv[1], h_qkv[2]

            h_attn = (h_q @ h_k.transpose(-2, -1)) * self.scale
            h_attn = h_attn.softmax(dim=-1)
            h_attn = (h_attn @ h_v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
            h_x = h_attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim).permute(0, 3, 1, 2)

            h_x = self.h_proj(h_x).permute(0, 2, 3, 1)


        if self.h_heads!=0 and self.l_heads!=0:
            out = torch.cat([l_x, h_x], dim=-1)
            out = out.reshape(B, N, C)

        if self.l_heads==0:
            out = h_x.reshape(B, N, C)

        if self.h_heads==0:
            out = l_x.reshape(B, N, C)

        return out

    def forward(self, x):
        return self.hi_lofi(x)

class FFN1(nn.Module):
    def __init__(self, dim, h_dim=None, out_dim=None):
        super().__init__()
        self.h_dim = dim*2 if h_dim==None else h_dim
        self.out_dim = dim if out_dim==None else out_dim

        self.act = nn.GELU()
        self.fc1 = DSC(dim, self.h_dim)
        self.norm = nn.BatchNorm2d(self.out_dim)
        self.fc2 = DSC(self.h_dim, self.h_dim)
        self.fc3 = IDSC(self.h_dim, self.out_dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N**0.5)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = self.act(self.fc3(self.act(self.fc2(self.act(self.fc1(x))))))
        x = self.norm(x).reshape(B, C, -1).permute(0, 2, 1)

        return x

class Block1(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=2, alpha=0.5, qkv_bias=False, qk_scale=None, h_dim=None, out_dim=None):
        super().__init__()
        self.hilo = FCHiLo1(dim, num_heads, qkv_bias, qk_scale, window_size, alpha)
        self.ffn = FFN1(dim, h_dim, out_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    def forward(self, x):
        x = x + self.norm1(self.hilo(x))
        x = x + self.norm2(self.ffn(x))
        return x

class Backbone(nn.Module):
    def __init__(self, inc=3, outc=96, resolution=512, patchsz=8):
        super().__init__()
        self.embedding = PatchEmbedding(inc, resolution, outc, patchsz)
        self.block1 = Block1(outc, 8, 8, 0.4)
        self.block12 = Block1(outc, 8, 8, 0.4)
        self.block13 = Block1(outc, 8, 8, 0.4)
        self.merge1 = PatchMerge(outc, 2)

        self.block2 = Block1(outc*2, 8, 4, 0.3)
        self.block22 = Block1(outc*2, 8, 4, 0.3)
        self.block23 = Block1(outc*2, 8, 4, 0.3)
        self.merge2 = PatchMerge(outc*2, 2)

        self.block3 = Block1(outc*4, 8, 4, 0.2)
        self.block32 = Block1(outc * 4, 8, 4, 0.2)
        self.block33 = Block1(outc * 4, 8, 4, 0.2)
        self.merge3 = PatchMerge(outc*4, 2)

        self.block4 = Block1(outc*8, 8, 4, 0.1)
        self.block42 = Block1(outc * 8, 8, 4, 0.1)
        self.block43 = Block1(outc * 8, 8, 4, 0.1)

    def forward(self, x):
        x = self.embedding(x)
        x1 = self.block1(x)
        x1 = self.block12(x1)
        x1 = self.block13(x1)

        x2 = self.merge1(x1)
        x2 = self.block2(x2)
        x2 = self.block22(x2)
        x2 = self.block23(x2)

        x3 = self.merge2(x2)
        x3 = self.block3(x3)
        x3 = self.block32(x3)
        x3 = self.block33(x3)

        x4 = self.merge3(x3)
        out = self.block4(x4)
        out = self.block42(out)
        out = self.block43(out)

        return x1, x2, x3, out

