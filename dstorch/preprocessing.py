import os

import cv2
import numpy as np
import pandas as pd
import scipy.ndimage as nd
from tqdm import tqdm

from dstorch import io


def label2distance(label):
    distance = np.zeros_like(label, dtype=float)
    for i in np.unique(label):
        if i:
            instance = label == i
            dt = nd.distance_transform_edt(instance)
            dt /= dt.max()
            distance += dt
    return distance


def get_distances(labels):
    distances = []
    for label in tqdm(labels, desc="Get distances"):
        distance = label2distance(label)
        distances.append(distance)
    return distances


def invert_images(images: list):
    inverts = []
    for image, _id in zip(images):
        if np.mean(image[..., 0]) > 100:
            inverts.append(255 - image)
        else:
            inverts.append(image)
    return inverts


def pixels_size(x):
    return sum([int(x) for x in x.split(' ')[1::2]])

def get_scale_df(ids, images, labels_file, size):
    labels = pd.read_csv(labels_file)
    labels['area'] = labels['EncodedPixels'].apply(pixels_size)
    df_area = labels.groupby('ImageId')['area'].median().reset_index()
    df_shape = pd.DataFrame(
        [[_id, *img.shape[:2]] for _id, img in zip(ids, images)],
        columns=['ImageId', 'height', 'width']
    )
    df_scale = df_shape.merge(df_area, on='ImageId', how='left')
    df_scale['scale_width'] = (df_scale['width'] * (size / df_scale['area']) ** 0.5).astype(int)
    df_scale['scale_height'] = (df_scale['height'] * (size / df_scale['area']) ** 0.5).astype(int)
    return df_scale

def prepare_data(
        train_path, test_path, extra_path, output_path,
        invert, center_thresholds, contour_thresholds, scale, labels_file, test_labels_file, size):
    ids, images, masks, labels = io.read_train_data(train_path)
    test_ids, test_images = io.read_test_data(test_path)
    extra_ids, extra_images, extra_masks, extra_labels = io.read_train_data(extra_path)

    ids.extend(extra_ids)
    images.extend(extra_images)
    masks.extend(extra_masks)
    labels.extend(extra_labels)

    if invert:
        images = invert_images(images)
        test_images = invert_images(test_images)

    distances = get_distances(labels)

    centers = []
    for threshold in center_thresholds:
        centers.append([dist >= threshold for dist in distances])

    contours = []
    for threshold in contour_thresholds:
        contours.append([(dist < threshold) * mask for dist, mask in zip(distances, masks)])

    answers = [np.stack(answer, axis=-1) for answer in zip(masks, *centers, *contours)]

    path_images = os.path.join(output_path, "images")
    os.makedirs(path_images, exist_ok=True)
    path_masks = os.path.join(output_path, "masks")
    os.makedirs(path_masks, exist_ok=True)
    path_distances = os.path.join(output_path, "distances")
    os.makedirs(path_distances, exist_ok=True)
    path_test = os.path.join(output_path, "test")
    os.makedirs(path_test, exist_ok=True)

    for image, mask, dist, _id in tqdm(zip(images, answers, distances, ids), total=len(ids), desc='Save train data'):
        cv2.imwrite(os.path.join(path_images, "{}.png".format(_id)), image)
        cv2.imwrite(os.path.join(path_masks, "{}.png".format(_id)), dist)
        cv2.imwrite(os.path.join(path_distances, "{}.png".format(_id)), mask)

    for image, _id in tqdm(zip(test_images, test_ids), total=len(test_ids), desc='Save test data'):
        cv2.imwrite(os.path.join(path_test, "{}.png".format(_id)), image)

    if scale:
        path_images = os.path.join(output_path, "scaled_images")
        os.makedirs(path_images, exist_ok=True)
        path_masks = os.path.join(output_path, "scaled_masks")
        os.makedirs(path_masks, exist_ok=True)
        path_test = os.path.join(output_path, "scaled_test")
        os.makedirs(path_test, exist_ok=True)
        
        df_scale = get_scale_df(ids, images, labels_file, size)
        for _id, img, answer in tqdm(zip(ids, images, answers), total=len(ids), desc='Scale & save train data'):
            height = df_scale.loc[df_scale['ImageId'] == _id, 'scale_height'].iloc[0]
            width = df_scale.loc[df_scale['ImageId'] == _id, 'scale_width'].iloc[0]

            cv2.imwrite(os.path.join(path_images, "{}.png".format(_id)), cv2.resize(img, (width, height)))
            cv2.imwrite(os.path.join(path_masks, "{}.png".format(_id)), cv2.resize(answer, (width, height)))

        df_scale_test = get_scale_df(test_ids, test_images, test_labels_file, size)
        for _id, img in tqdm(zip(test_ids, test_images), total=len(test_ids), desc='Scale & save test data'):
            height = df_scale_test.loc[df_scale_test['ImageId'] == _id, 'scale_height'].iloc[0]
            width = df_scale_test.loc[df_scale_test['ImageId'] == _id, 'scale_width'].iloc[0]

            cv2.imwrite(os.path.join(path_test, "{}.png".format(_id)), cv2.resize(img, (width, height)))
