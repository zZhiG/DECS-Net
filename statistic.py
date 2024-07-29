import numpy as np


def compute_confusion_matrix(precited,expected):
    # part = precited ^ expected
    part = np.logical_xor(precited, expected)
    pcount = np.bincount(part)
    # tp_list = list(precited & expected)
    # fp_list = list(precited & ~expected)
    tp_list = list(np.logical_and(precited, expected))
    fp_list = list(np.logical_and(precited, np.logical_not(expected)))
    tp = tp_list.count(1)
    fp = fp_list.count(1)
    tn = pcount[0] - tp
    fn = pcount[1] - fp
    return tp, fp, tn, fn


def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)
    precision = tp / (tp+fp)
    recall = tp / (tp+fn)
    F1 = (2*precision*recall) / (precision+recall)
    return accuracy, precision, recall, F1

def statistic(ipath, num):
    tp, fp, tn, fn = 0, 0, 0, 0
    for i in range(num):
        img_path = ipath + '/pred' + str(i + 1) + '.npy'
        mask_path = ipath + '/label' + str(i + 1) + '.npy'
        img = np.load(img_path)
        msk = np.load(mask_path)
        img[img > 0] = 1.
        img[img <= 0] = 0.
        msk[msk > 0] = 1.
        msk[msk <= 0] = 0.
        # print(msk.shape)
        img = img.reshape(-1)
        msk = msk.reshape(-1)

        tp_, fp_, tn_, fn_ = compute_confusion_matrix(img, msk)
        tp = tp + tp_
        fp = fp + fp_
        tn = tn + tn_
        fn = fn + fn_
    accuracy, precision, recall, F1 = compute_indexes(tp, fp, tn, fn)
    return accuracy, precision, recall, F1

def miou(path, epoachs, total):
    mIoU = 0
    for i in range(epoachs):
        img_path = path + '/pred' + str(i + 1) + '.npy'
        mask_path = path + '/label' + str(i + 1) + '.npy'
        img = np.load(img_path)
        msk = np.load(mask_path)
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
    mIoU = mIoU / total

    return mIoU

if __name__ == '__main__':
    ipath = r'results\example'
    num = 161
    total = 322

    mIoU = miou(ipath, num, total)
    acc, pr, re, f1 = statistic(ipath, num)

    print(f'accuracy:{acc}, precision:{pr}, recall:{re}, F1-score:{f1}, mIoU:{mIoU}')
    # accuracy:0.9880591919703513, precision:0.7825764887721974, recall:0.7902087536121258, F1-score:0.7863741026223604, mIoU:0.5868952288005637

