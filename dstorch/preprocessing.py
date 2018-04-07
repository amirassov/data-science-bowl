import os

import cv2
import numpy as np
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

def prepare_data(train_path, test_path, extra_path, output_path, invert, center_thresholds, contour_thresholds):
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

    for image, mask, dist, _id in tqdm(zip(images, answers, distances, ids), total=len(ids)):
        cv2.imwrite(os.path.join(output_path, "images", "{}.png".format(_id)), image)
        cv2.imwrite(os.path.join(output_path, "distances", "{}.png".format(_id)), dist)
        cv2.imwrite(os.path.join(output_path, "masks", "{}.png".format(_id)), mask)

    for image, _id in tqdm(zip(test_images, test_ids), total=len(test_ids)):
        cv2.imwrite(os.path.join(output_path, "test", "{}.png".format(_id)), image)
