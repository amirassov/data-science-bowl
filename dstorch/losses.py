from torch import nn
from torch.nn import functional as F
import torch


def dice_loss(input, target):
    dice_target = (target == 1).float()
    dice_input = F.sigmoid(input)

    intersection = (dice_target * dice_input).sum() + 1
    union = dice_target.sum() + dice_input.sum() + 1
    return torch.log(2.0 * intersection) - torch.log(union)


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
        return self.bce(input, target) - self.dice(input, target)


class BCEDiceLossMulti(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.bce_dice = BCEDiceLoss()
        self.num_classes = num_classes
    
    def forward(self, input, target):
        loss = 0
        for cls in range(self.num_classes):
            loss += self.bce_dice(input[:, cls], target[:, cls])
        return loss / self.num_classes
