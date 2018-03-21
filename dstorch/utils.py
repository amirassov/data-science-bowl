import numpy as np
import torch
from torch.autograd import Variable
import cv2


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x

def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()

def pad_image(image, period):
    w, h = image.shape[:2]

    lr = np.ceil(h / period) * period - h
    left = int(np.ceil(lr / 2))
    right = int(lr - left)
    
    tb = np.ceil(w / period) * period - w
    top = int(np.ceil(tb / 2))
    bottom = int(tb - top)
    
    return cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_REFLECT
    ), top, left
