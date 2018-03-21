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
    
    def forward(self, input, target):
        bce = nn.modules.loss.BCEWithLogitsLoss(size_average=self.size_average)(input, target)
        dice = self.dice(input, target)
        return 0.5 * bce - dice
