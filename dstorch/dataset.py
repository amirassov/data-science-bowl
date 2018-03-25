import cv2
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from dstorch.utils import to_float_tensor, pad_image


class BowlDataset(Dataset):
    def __init__(self, filenames, path_image, path_mask, transform, mode, period):
        self.filenames = filenames
        self.path_image = path_image
        self.path_mask = path_mask
        self.transform = transform
        self.mode = mode
        self.period = period

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img = cv2.imread(self.path_image.format(filename))
        mask = cv2.imread(self.path_mask.format(filename))
        img, mask = self.transform(img, mask)

        if self.mode == 'train':
            return to_float_tensor(img), to_float_tensor(mask)
        elif self.mode == 'validation':
            pad_img, _, _ = pad_image(img, self.period)
            pad_mask, _, _ = pad_image(mask, self.period)
            return to_float_tensor(pad_img), to_float_tensor(pad_mask)
        elif self.mode == 'predict':
            pad_img, top, left = pad_image(img, self.period)
            height, width = img.shape[:2]
            return to_float_tensor(pad_img), str(filename), top, left, height, width
        else:
            raise TypeError('Unknown mode type!')

def make_loader(filenames,  path_image, path_mask,
                batch_size, transform, workers=1, shuffle=True, mode='train', period=128):
    return DataLoader(
        dataset=BowlDataset(filenames, path_image, path_mask, transform=transform, mode=mode, period=period),
        shuffle=shuffle,
        num_workers=workers,
        batch_size=batch_size,
        pin_memory=torch.cuda.is_available()
    )
