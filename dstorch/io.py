import os

import numpy as np
from skimage import io
from tqdm import tqdm
from skimage import img_as_bool, img_as_ubyte, img_as_float


def read_train_data(train_path, d=3):
    train_ids = next(os.walk(train_path))[1]
    train_images = []
    train_masks = []

    for id_ in tqdm(train_ids, desc='Reading train data..'):
        path = train_path + id_
        image = img_as_ubyte(io.imread(path + '/images/' + id_ + '.png')[:, :, :d])
        train_images.append(image)
        mask = None
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = img_as_bool(io.imread(path + '/masks/' + mask_file))
            if len(mask_.shape) > 2:
                mask_ = mask_[..., 0]
            if mask is None:
                mask = np.zeros_like(mask_)
            mask = np.maximum(mask, mask_)
        train_masks.append(img_as_float(mask))
    return train_ids, train_images, train_masks

def read_test_data(test_path, d=3):
    test_ids = next(os.walk(test_path))[1]
    test_images = []
    
    for id_ in tqdm(test_ids, desc='Reading test data..'):
        path = test_path + id_
        image = img_as_ubyte(io.imread(path + '/images/' + id_ + '.png')[:, :, :d])
        test_images.append(image)
    return test_ids, test_images


