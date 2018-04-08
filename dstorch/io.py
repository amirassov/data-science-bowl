import os

import numpy as np
import cv2
from tqdm import tqdm
from skimage import img_as_bool, img_as_ubyte, img_as_float
import pandas as pd


def read_test_data(test_path, scale_path=None, d=3):
    test_ids = next(os.walk(test_path))[1]
    test_images = []
    if scale_path:
        df_scale = pd.read_csv(scale_path)

    for id_ in tqdm(test_ids, desc='Reading test data..'):
        path = test_path + id_
        image = img_as_ubyte(cv2.imread(path + '/images/' + id_ + '.png')[:, :, :d])

        if scale_path:
            height = df_scale.loc[df_scale['ImageId'] == id_, 'scale_height'].iloc[0]
            width = df_scale.loc[df_scale['ImageId'] == id_, 'scale_width'].iloc[0]
            image = cv2.resize(image, (width, height), cv2.INTER_NEAREST)

        test_images.append(image)
    return test_ids, test_images


def read_train_data(train_path, scale_path=None, d=3):
    train_ids = next(os.walk(train_path))[1]
    train_images = []
    train_masks = []
    train_labels = []
    if scale_path:
        df_scale = pd.read_csv(scale_path)
        
    for id_ in tqdm(train_ids, desc='Reading train data..'):
        path = train_path + id_
        image_file = next(os.walk(path + '/images/'))[2][0]
        image = img_as_ubyte(cv2.imread(path + '/images/' + image_file)[:, :, :d])
        
        if scale_path:
            height = df_scale.loc[df_scale['ImageId'] == id_, 'scale_height'].iloc[0]
            width = df_scale.loc[df_scale['ImageId'] == id_, 'scale_width'].iloc[0]
            image = cv2.resize(image, (width, height), cv2.INTER_NEAREST)
        else:
            height = None
            width = None
        
        train_images.append(image)
        mask = None
        label = None
        for i, mask_file in enumerate(next(os.walk(path + '/masks/'))[2]):
            mask_ = img_as_float(cv2.imread(path + '/masks/' + mask_file))

            if scale_path:
                mask_ = cv2.resize(mask_, (width, height), cv2.INTER_NEAREST)

            if len(mask_.shape) > 2:
                mask_ = mask_[..., 0]
            if mask is None:
                mask = np.zeros_like(mask_)

            if label is None:
                label = np.zeros_like(mask_, dtype=float)

            mask = np.maximum(mask, mask_)
            label += (i + 1) * mask_
        train_masks.append(img_as_float(mask))
        train_labels.append(img_as_ubyte(label))
    return train_ids, train_images, train_masks, train_labels
