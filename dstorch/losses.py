import torch
from torch import nn
from torch.nn import functional as F


def dice_loss(preds, trues, is_average=True):
    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    intersection = (preds * trues).sum(1)
    scores = 2.0 * (intersection + 1) / (preds.sum(1) + trues.sum(1) + 1)

    if is_average:
        score = scores.sum() / num
        return torch.clamp(score, 0., 1.)
    else:
        return scores


class DiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
    
    def forward(self, input, target):
        return dice_loss(F.sigmoid(input), target, is_average=self.size_average)


class BCEDiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.dice = DiceLoss(size_average=size_average)
        self.bce = nn.modules.loss.BCEWithLogitsLoss(size_average=self.size_average)
    
    def forward(self, input, target):
        return 0.5 * self.bce(input, target) - self.dice(input, target)


class BCEDiceLossMulti(nn.Module):
    def __init__(self, size_average=True, num_classes=1):
        super().__init__()
        self.size_average = size_average
        self.bce_dice = BCEDiceLoss(size_average=size_average)
        self.num_classes = num_classes

    def forward(self, input, target):
        loss = 0
        for cls in range(self.num_classes):
            loss += self.bce_dice(input[:, cls], target[:, cls])
        return loss / self.num_classes
