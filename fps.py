import time
import torch
import numpy as np

from src.DECSNet import DECSNet


Net = DECSNet(3, 64, 512, 4, 'resnet50', False, [256, 512, 1024, 2048], [64, 128, 256, 512], 1).cuda()
ckp = torch.load('c3238_1.pth')
Net.load_state_dict(ckp['net'])
Net.eval()

'''
img_list = [---Save test image, omit here, you can add it yourself---]
'''

time_ = []

for i in range(100):
    t1 = time.time()
    Net(img_list[i])
    t2 = time.time()
    time_.append(t2 - t1)

print('average time:', np.mean(time_) / 1)
print('average fps:', 1 / np.mean(time_))

print('fastest time:', min(time_) / 1)
print('fastest fps:', 1 / min(time_))

print('slowest time:', max(time_) / 1)
print('slowest fps:', 1 / max(time_))
