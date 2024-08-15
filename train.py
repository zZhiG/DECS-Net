import os.path
import torch
from tensorboardX import SummaryWriter
from torch.nn import functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.data import DataLoader
from src.DECSNet import DECSNet
from datasets.loader import DatasetLoader


def computer_loss(pred, truth):
    p = pred.view(-1, 1)
    t = truth.view(-1, 1)
    loss1 = F.binary_cross_entropy_with_logits(p, t, reduction='mean')

    num = truth.size(0)
    smooth = 1
    probs = torch.sigmoid(pred)
    m1 = probs.view(num, -1)
    m2 = truth.view(num, -1)
    intersection = (m1 * m2)
    score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
    loss2 = 1 - score.sum() / num

    return loss1 + loss2


def compute_confusion_matrix(precited, expected):
    precited = precited.detach().cpu().numpy()
    expected = expected.detach().cpu().numpy()
    part = np.logical_xor(precited, expected)
    pcount = np.bincount(part)
    tp_list = list(np.logical_and(precited, expected))
    fp_list = list(np.logical_and(precited, np.logical_not(expected)))
    tp = tp_list.count(1)
    fp = fp_list.count(1)
    tn = pcount[0] - tp
    fn = pcount[1] - fp
    return tp, fp, tn, fn


def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = (2 * precision * recall) / (precision + recall)
    return accuracy, precision, recall, F1


def miou(img, msk):
    mIoU = 0
    img[img > 0] = 1.
    img[img <= 0] = 0.
    msk[msk > 0] = 1.
    msk[msk <= 0] = 0.
    B = img.shape[0]
    for j in range(B):
        pred = img[j].reshape(-1)
        label = msk[j].reshape(-1)
        tp_, fp_, tn_, fn_ = compute_confusion_matrix(pred, label)
        iou1 = tp_ / (tp_ + fp_ + fn_)
        mIoU += iou1
    return mIoU


def train():
    device = 'cuda'
    Net = DECSNet(3, 64, 512, 4, 'resnet50', True, [256, 512, 1024, 2048], [64, 128, 256, 512], 1).to(device)

    resume = False
    epoch = 300
    save_epoch = 5
    save_weight_path = 'weights'
    if not os.path.exists(save_weight_path):
        os.makedirs(save_weight_path)
    save_weight_name = 'example.pth'
    save_weight = os.path.join(save_weight_path, save_weight_name)

    initial_lr = 0.001
    batchsz = 24

    optimizer = optim.AdamW(Net.parameters(), lr=initial_lr)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=350, T_mult=2)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200)

    writer = SummaryWriter(logdir='xxxx')

    filepath = r'E:\dataset\Crack3238'
    d = DatasetLoader(filepath, 'train')
    train_loader = DataLoader(d, batch_size=batchsz, shuffle=True)

    start_epoch = 0
    if resume:
        path_checkpoint = r"xxxx"
        checkpoint = torch.load(path_checkpoint)
        Net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    total_step = 0
    for it in range(epoch):
        it = start_epoch + it
        if it == (epoch - 1):
            break
        Net.train()

        avg_loss = 0
        now_lr = optimizer.param_groups[0]['lr']

        for idx, (img, lab) in enumerate(train_loader):
            total_step += 1
            img = img.to(device)
            lab = lab.to(device)

            optimizer.zero_grad()
            output = Net(img)
            loss = computer_loss(output, lab)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            writer.add_scalar('Train/Loss', loss.item(), total_step)

        print('epoch:{}--avg loss:{}--lr:{}'.format(it, avg_loss / total_step, now_lr))

        if it % save_epoch == 0:
            Net.eval()
            print('=====================val and Save===================')
            mIoU = 0
            tp, tn, fp, fn = 0, 0, 0, 0
            for idx, (img, lab) in enumerate(train_loader):
                img = img.to(device)
                lab = lab.to(device)
                output = Net(img)
                mIoU += miou(output, lab)
                tp_, fp_, tn_, fn_ = compute_confusion_matrix(output.reshape(-1), lab.reshape(-1))
                tp = tp + tp_
                fp = fp + fp_
                tn = tn + tn_
                fn = fn + fn_
            accuracy, pr, re, f1 = compute_indexes(tp, fp, tn, fn)

            print(f'it:{it}, mIoU:{mIoU / d.num_of_samples()}, pr{pr}, re{re}, f1{f1}')

            writer.add_scalar('Val/Precision', pr, it)
            writer.add_scalar('Val/Recall', re, it)
            writer.add_scalar('Val/mIoU', mIoU / d.num_of_samples(), it)
            writer.add_scalar('Val/F1-score', f1, it)
            writer.add_scalar('Val/Accuracy', accuracy, it)
            print('=====================End===================')

            checkpoint_mid = {
                "net": Net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": it
            }

            torch.save(checkpoint_mid, f'weights/{it}.pth')

        scheduler.step()

    checkpoint = {
        "net": Net.state_dict(),
        'optimizer': optimizer.state_dict(),
        "epoch": it
    }

    torch.save(checkpoint, save_weight)


if __name__ == '__main__':
    train()
