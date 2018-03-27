import numpy as np
import scipy.ndimage as nd
from skimage import img_as_ubyte
from skimage import morphology
from skimage.filters import gaussian


def remove_smalls(mask, width_holes, width_objects):
    remove_holes = morphology.remove_small_holes(
        mask.astype(bool),
        min_size=width_holes ** 3
    )

    remove_objects = morphology.remove_small_objects(
        remove_holes.astype(bool),
        min_size=width_objects ** 3
    )
    return remove_holes, remove_objects


def distance_transform(mask, sigma):
    distance = nd.distance_transform_edt(mask)
    distance = gaussian(distance, sigma=sigma)
    return distance


def pad_zero(image, padding_size, padding=True):
    h, w = image.shape
    if padding:
        padded_image = np.zeros((h + 2 * padding_size, w + 2 * padding_size), dtype=image.dtype)
        padded_image[padding_size:padding_size + h, padding_size:padding_size + w] = image
        return padded_image
    else:
        return image[padding_size:-padding_size, padding_size:-padding_size]


def watershed(
        prediction,
        mask_threshold=0.5, center_threshold=0.7, width_holes=4,
        width_objects=3, padding_size=5, min_size=16
):
    pred_mask = prediction[..., 0]
    pred_center = prediction[..., 1]

    mask = img_as_ubyte(pred_mask > mask_threshold)
    center = img_as_ubyte(pred_center > center_threshold)
    _, mask = remove_smalls(mask, width_holes, width_objects)

    padded_mask = pad_zero(mask, padding_size=padding_size)
    padded_center = pad_zero(center, padding_size=padding_size)

    markers = nd.label(padded_center)[0]

    watershed_image = pad_zero(
        morphology.watershed(-padded_mask, markers, mask=padded_mask),
        padding_size=padding_size, padding=False
    )

    unique, counts = np.unique(watershed_image, return_counts=True)
    for (k, v) in dict(zip(unique, counts)).items():
        if v < min_size:
            watershed_image[watershed_image == k] = 0
    return watershed_image


def get_labels(labeled_image):
    labels = []
    for i in np.unique(labeled_image):
        if i:
            mask = labeled_image == i
            labels.append(mask)
    return np.array(labels)
