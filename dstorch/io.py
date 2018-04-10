import os

import numpy as np
import cv2
from tqdm import tqdm
from skimage import img_as_bool, img_as_ubyte, img_as_float


def read_test_data(test_path, d=3):
    test_ids = next(os.walk(test_path))[1]
    test_images = []

    for id_ in tqdm(test_ids, desc='Reading test data..'):
        path = test_path + id_
        image = img_as_ubyte(cv2.imread(path + '/images/' + id_ + '.png')[:, :, :d])
        test_images.append(image)
    return test_ids, test_images


def read_train_data(train_path, d=3):
    train_ids = next(os.walk(train_path))[1]
    train_images = []
    train_masks = []
    train_labels = []

    for id_ in tqdm(train_ids, desc='Reading train data..'):
        path = train_path + id_
        image_file = next(os.walk(path + '/images/'))[2][0]
        image = img_as_ubyte(cv2.imread(path + '/images/' + image_file)[:, :, :d])
        train_images.append(image)
        mask = None
        label = None
        for i, mask_file in enumerate(next(os.walk(path + '/masks/'))[2]):
            mask_ = img_as_bool(cv2.imread(path + '/masks/' + mask_file))
            if len(mask_.shape) > 2:
                mask_ = mask_[..., 0]
            if mask is None:
                mask = np.zeros_like(mask_)

            if label is None:
                label = np.zeros_like(mask_, dtype=int)

            mask = np.maximum(mask, mask_)
            label += (i + 1) * mask_
        train_masks.append(img_as_float(mask))
        train_labels.append(label)
    return train_ids, train_images, train_masks, train_labels
