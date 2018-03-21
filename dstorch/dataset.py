import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dstorch.utils import to_float_tensor, pad_image


class BowlDataset(Dataset):
    def __init__(self, images, masks=None, ids=None, transform=None, mode='train'):
        self.images = images
        self.masks = masks
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.masks is not None:
            mask = self.masks[idx]
        else:
            mask = None

        img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        elif self.mode == 'validation':
            pad_img, _, _ = pad_image(img, 128)
            pad_mask, _, _ = pad_image(np.expand_dims(mask, 0), 128)
            return to_float_tensor(pad_img), torch.from_numpy(pad_mask).float()
        elif self.mode == 'predict':
            pad_img, top, left = pad_image(img, 128)
            return to_float_tensor(pad_img), str(self.ids[idx]), top, left
        else:
            raise TypeError('Unknown mode type!')


def make_loader(images, masks, ids, batch_size, transform, workers=1, shuffle=True, mode='train'):
    if images is None:
        raise TypeError('Images must be a list!')
    if ids is None:
        raise TypeError('Ids must be a list!')

    return DataLoader(
        dataset=BowlDataset(images, masks, ids, transform=transform, mode=mode),
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
