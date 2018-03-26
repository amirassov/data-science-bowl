from torch import nn
from torch.nn import functional as F
import torch


def dice_loss(input, target):
    EPS = 1.0
    dice_target = (target == 1).float()
    dice_input = F.sigmoid(input)
    
    intersection = (dice_target * dice_input).sum() + EPS
    union = dice_target.sum() + dice_input.sum() + EPS
    return 2.0 * intersection / union


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input, target):
        return dice_loss(input, target)


class BCEDiceLoss(nn.Module):
    def __init__(self, size_average=True):
        super().__init__()
        self.size_average = size_average
        self.dice = DiceLoss()
        self.bce = nn.modules.loss.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce(input, target) - torch.log(self.dice(input, target))


class BCEDiceLossCenters(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.bce_dice = BCEDiceLoss()
        self.weights = weights

    def forward(self, input, target):
        mask_input, mask_target = input[:, 0], target[:, 0]
        center_input, center_target = input[:, 1], target[:, 1]
        center_input_04, center_target_04 = input[:, 2], target[:, 2]
    
        loss = self.weights['mask'] * self.bce_dice(mask_input, mask_target)
        loss += self.weights['center'] * self.bce_dice(center_input, center_target)
        loss += self.weights['center_04'] * self.bce_dice(center_input_04, center_target_04)
        return loss
