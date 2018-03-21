import pandas as pd


def invert_images(classes: pd.DataFrame, images: list, ids: list):
    inverts = []
    for image, _id in zip(images, ids):
        if classes.loc[classes['id'] == _id, 'background'].iloc[0] in ['white', 'yellow']:
            inverts.append(255 - image)
        else:
            inverts.append(image)
    return inverts


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


