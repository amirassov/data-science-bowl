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
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.modules.loss.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce(input, target) - torch.log(self.dice(input, target))


class BCEDiceLossWithoutLog(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.modules.loss.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce(input, target) - self.dice(input, target)


class BCEDiceLossOneClass(nn.Module):
    def __init__(self, cls):
        super().__init__()
        self.bce_dice = BCEDiceLoss()
        self.cls = cls

    def forward(self, input, target):
        return self.bce_dice(input[:, self.cls], target[:, self.cls])


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


class BCEDiceLossMulti(nn.Module):
    def __init__(self, num_classes, weights):
        super().__init__()
        self.bce_dice = BCEDiceLoss()
        self.num_classes = num_classes
        self.weights = weights

    def forward(self, input, target):
        loss = 0
        for cls in range(self.num_classes):
            channel_input, channel_target = input[:, cls], target[:, cls]
            loss += self.weights[cls] * self.bce_dice(channel_input, channel_target)
        return loss


class BCEDiceLossMultiWithoutLog(nn.Module):
    def __init__(self, num_classes, weights):
        super().__init__()
        self.bce_dice = BCEDiceLossWithoutLog()
        self.num_classes = num_classes
        self.weights = weights

    def forward(self, input, target):
        loss = 0
        for cls in range(self.num_classes):
            channel_input, channel_target = input[:, cls], target[:, cls]
            loss += self.weights[cls] * self.bce_dice(channel_input, channel_target)
        return loss
