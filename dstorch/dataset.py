import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from dstorch.utils import to_float_tensor, pad_image


class BowlDataset(Dataset):
    def __init__(self, images, masks, ids, transform, mode, period):
        self.images = images
        self.masks = masks
        self.ids = ids
        self.transform = transform
        self.mode = mode
        self.period = period

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        mask = None if self.masks is None else self.masks[index]
        img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        elif self.mode == 'validation':
            pad_img, top, left = pad_image(img, self.period)
            pad_mask = np.expand_dims(pad_image(mask, self.period)[0], 0)
            return to_float_tensor(pad_img), torch.from_numpy(pad_mask).float(), top, left
        elif self.mode == 'predict':
            pad_img, top, left = pad_image(img, self.period)
            return to_float_tensor(pad_img), str(self.ids[index]), top, left
        else:
            raise TypeError('Unknown mode type!')

def make_loader(images, masks, ids, batch_size, transform, workers=1, shuffle=True, mode='train', period=128):
    if images is None:
        raise TypeError('Images must be a list!')
    if ids is None:
        raise TypeError('Ids must be a list!')

    return DataLoader(
        dataset=BowlDataset(images, masks, ids, transform=transform, mode=mode, period=period),
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
