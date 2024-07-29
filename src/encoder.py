import torch
import torch.nn as nn
from transformer import Backbone
from resnet import *
from ffm import FFM


class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel, resolution, patchsz, mode, pretrained, cnn_dim, transform_dim):
        super().__init__()
        self.transformer = Backbone(in_channel, out_channel, resolution, patchsz)

        self.cnn = get_resnet(mode, pretrained)
        self.init_CNN = nn.Sequential(self.cnn.conv1,
                                      self.cnn.bn1,
                                      self.cnn.relu)
        self.maxp = self.cnn.maxpool
        self.layer1 = self.cnn.layer1
        self.layer2 = self.cnn.layer2
        self.layer3 = self.cnn.layer3
        self.layer4 = self.cnn.layer4
        #
        self.ffm1 = FFM(cnn_dim[0], transform_dim[0], 128)
        self.ffm2 = FFM(cnn_dim[1], transform_dim[1], 64)
        self.ffm3 = FFM(cnn_dim[2], transform_dim[2], 32)
        self.ffm4 = FFM(cnn_dim[3], transform_dim[3], 16)

    def forward(self, x):
        # cnn
        cnn = self.init_CNN(x)
        cnn = self.maxp(cnn)
        cnn1 = self.layer1(cnn)
        cnn2 = self.layer2(cnn1)
        cnn3 = self.layer3(cnn2)
        cnn4 = self.layer4(cnn3)

        # transformer
        t1, t2, t3, out = self.transformer(x)

        o1 = self.ffm1(cnn1, t1)
        o2 = self.ffm2(cnn2, t2)
        o3 = self.ffm3(cnn3, t3)
        o4 = self.ffm4(cnn4, out)

        return o1, o2, o3, o4


if __name__ == '__main__':
    x = torch.rand(2, 3, 256, 256).cuda()
    net = Encoder(3, 64, 256, 4, 'resnet34', True, [64, 128, 256, 512], [64, 128, 256, 512]).cuda()
    out = net(x)

