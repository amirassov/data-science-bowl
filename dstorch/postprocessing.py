import numpy as np
import scipy.ndimage as nd
from skimage import img_as_ubyte
from skimage import morphology


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
        mask_threshold=0.7,
        center_threshold=0.7,
        padding_size=5,
        min_size=50,
):
    pred_mask = prediction[..., 0]
    pred_center = prediction[..., 1]

    mask = img_as_ubyte(pred_mask > mask_threshold)
    center = img_as_ubyte(pred_center > center_threshold)

    markers = nd.label(pad_zero(center, 5))[0]

    watershed_image = pad_zero(
        morphology.watershed(
            -pad_zero(pred_mask, padding_size=padding_size),
            markers,
            mask=pad_zero(mask, padding_size=padding_size)
        ),
        padding_size=padding_size, padding=False
    )

    unique, counts = np.unique(watershed_image, return_counts=True)
    for (k, v) in dict(zip(unique, counts)).items():
        if v < min_size:
            watershed_image[watershed_image == k] = 0
    return watershed_image
