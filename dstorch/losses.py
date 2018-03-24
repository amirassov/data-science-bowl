from torch import nn
from torch.nn import functional as F
import torch


def dice_loss(input, target):
    EPS = 1e-15
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
        return 0.5 * self.bce(input, target) - self.dice(input, target)


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


class BCEDiceLossMCC(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_dice = BCEDiceLoss()
    
    def forward(self, input, target):
        mask_input, mask_target = input[:, 0], target[:, 0]
        contour_input, contour_target = input[:, 1], target[:, 1]
        center_input, center_target = input[:, 2], target[:, 2]
        
        loss = 0.3 * self.bce_dice(mask_input, mask_target)
        loss += 0.2 * self.bce_dice(contour_input, contour_target)
        loss += 0.3 * self.bce_dice(center_input, center_target)
        loss += 0.1 * self.bce_dice(torch.mul(mask_input, (1.0 - contour_input)), center_target)
        loss += 0.1 * self.bce_dice(torch.mul(mask_input, (1.0 - center_input)), contour_target)
        return loss / (0.3 + 0.2 + 0.3 + 0.1 + 0.1)
