import torch
from thop import profile, clever_format

from src.DECSNet import DECSNet


Net = DECSNet(3, 64, 512, 4, 'resnet50', False, [256, 512, 1024, 2048], [64, 128, 256, 512], 1)
input = torch.randn(1, 3, 256, 256)

macs, params = profile(Net, (input,))
macs, params = clever_format([macs, params], "%.3f")

print('MACs: ', macs, 'params: ', params)
# MACs:  26.085G params:  68.865M
