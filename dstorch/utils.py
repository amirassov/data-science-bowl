import cv2
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from tqdm import tqdm


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
        cv2.BORDER_REFLECT_101
    ), top, left


def make_submission(fname, ids, rles):
    sub = pd.DataFrame()
    sub['ImageId'] = ids
    sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
    sub.to_csv(fname, index=False)
    return sub


def train_val_split(classes: pd.DataFrame, ids: list, *args):
    train_id_set = set(classes.loc[classes['type'] == 'train', 'id'])
    val_id_set = set(classes.loc[classes['type'] == 'val', 'id'])

    train_indices = [i for i, image_id in enumerate(ids) if image_id in train_id_set]
    val_indices = [i for i, image_id in enumerate(ids) if image_id in val_id_set]

    train_args = []
    val_args = []
    for arg in args:
        train_args.append([arg[i] for i in train_indices])
        val_args.append([arg[i] for i in val_indices])
    return train_args, val_args


def rle_decoding(rle, shape):
    h, w = shape[:2]
    s = np.asarray(rle.split(), dtype=int)
    starts, lengths = s[::2] - 1, s[1::2]
    ends = starts + lengths
    img = np.zeros(h * w, dtype=np.uint8)
    for start, end in zip(starts, ends):
        img[start:end] = 1
    return img.reshape(w, h).T.reshape(shape)


def rles2labels(rles, shapes, ids):
    labels = []
    for image_id, shape in tqdm(zip(ids, shapes), total=len(ids)):
        group = rles[rles['ImageId'] == image_id]
        label = np.zeros(shape, dtype=np.uint8)
        for i, (_, row) in enumerate(group.iterrows()):
            label += (i + 1) * rle_decoding(row['EncodedPixels'], shape)
        labels.append(label)
    return labels


def rle_encode(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def probability2rle(probability, postprocess_function, **kwargs):
    labeled_image = postprocess_function(probability, **kwargs)
    for i in np.unique(labeled_image):
        if i:
            yield rle_encode(labeled_image == i)


def mask2rle(predictions, ids, postprocess_function, **kwargs):
    mask_ids = []
    rles = []
    for i, id_ in tqdm(enumerate(ids), total=len(ids)):
        rle = list(probability2rle(predictions[i], postprocess_function, **kwargs))
        rles.extend(rle)
        mask_ids.extend([id_] * len(rle))
    return mask_ids, rles
