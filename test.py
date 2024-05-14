import torch
import numpy as np

from src.DECSNet import DECSNet
from datasets.loader import dataloader
from torch.utils.data import DataLoader


Net = DECSNet(3, 64, 512, 4, 'resnet50', False, [256, 512, 1024, 2048], [64, 128, 256, 512], 1).cuda()

Net.eval()
ckp = torch.load('example.pth')
Net.load_state_dict(ckp['net'])

d = dataloader()
train_loader = DataLoader(d, batch_size=2, shuffle=False)

for idx, (img, lab) in enumerate(train_loader):
    img = img.cuda()
    lab = lab.cuda()

    output = Net(img)

    np.save(r'results/example/pred' + str(idx + 1) + '.npy', output.detach().cpu().numpy())
    np.save(r'results/example/label' + str(idx + 1) + '.npy', lab.detach().cpu().numpy())