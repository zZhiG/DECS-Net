import torch
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFilter
import matplotlib.pylab as plt
import random
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DatasetLoader(Dataset):
    def __init__(self, filepath='E:\dataset\Crack3238', mode='test', h=256, w=256):
        super().__init__()
        self.filepath = filepath

        self.h = h
        self.w = w

        self.images = []
        self.labels = []

        namepath = mode+'.txt'

        with open(namepath, 'r') as f:
            names = f.read().split('\n')

        for i in names:
            if i == '':
                break
            self.images.append(os.path.join(filepath, 'img', i))
            self.labels.append(os.path.join(filepath, 'labelcol', i))

    def __len__(self):
        return len(self.images)

    def num_of_samples(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            image = self.images[idx]
            mask = self.labels[idx]
        else:
            image = self.images[idx]
            mask = self.labels[idx]

        image = Image.open(image)
        mask = Image.open(mask)
        tf = transforms.Compose([
            transforms.Resize((int(self.h), int(self.w))),
            transforms.ToTensor()
        ])

        image = image.convert('RGB')
        norm = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        set_seed(1234)
        img = tf(image)
        img = norm(img)

        set_seed(1234)
        mask = tf(mask)
        mask[mask > 0] = 1

        return (img, mask)


