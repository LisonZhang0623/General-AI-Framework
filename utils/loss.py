import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


# BCE and IOU Loss
def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')  # BCE
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    # iou
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


# 骰子损失 处理正负样本之间的强烈不平衡
def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


class dice_bce_loss(nn.Module):
    def __init__(self, batch=False):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def dice_metric(self, y_true, y_pred):
        dice = self.soft_dice_coeff(y_true, y_pred)
        return dice

    def resize(self, y_true, h, w):
        b = y_true.shape[0]
        y = np.zeros((b, h, w, y_true.shape[1]))

        y_true = np.array(y_true.cpu())
        for id in range(b):
            y1 = y_true[id, :, :, :].transpose(1, 2, 0)
            a = cv2.resize(y1, (h, w))
            if a.ndim == 2:
                a = np.expand_dims(a, axis=-1)
            y[id, :, :, :] = a
        y = y.transpose(0, 3, 1, 2)
        return torch.Tensor(y)

    def __call__(self, y_pred, y_true):
        # the ground_truth map is resized to the resolution of the predicted map during training
        if y_true.shape[2] != y_pred.shape[2] or y_true.shape[3] != y_pred.shape[3]:
            y_true = self.resize(y_true, y_pred.shape[2], y_pred.shape[3]).cuda()
        # a = self.bce_loss(y_pred, y_true)
        a = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')
        y_pred = torch.sigmoid(y_pred)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


def criterion(inputs, target):
    loss = structure_loss(inputs, target)
    return loss
