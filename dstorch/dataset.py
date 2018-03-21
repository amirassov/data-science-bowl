import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def to_float_tensor(img):
        return torch.from_numpy(np.moveaxis(img, -1, 0)).float()

class BowlDataset(Dataset):
    def __init__(self, images: list, masks: list, ids: list, transform, mode='train'):
        self.images = images
        self.masks = masks
        self.ids = ids
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return to_float_tensor(img), torch.from_numpy(np.expand_dims(mask, 0)).float()
        elif self.mode == 'test':
            return to_float_tensor(img), str(self.ids[idx])
        else:
            print('Unknown mode type!')


def make_loader(images, masks, ids, batch_size, workers, shuffle, transform, mode):
    return DataLoader(
        dataset=BowlDataset(images, masks, ids, transform=transform, mode=mode),
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
