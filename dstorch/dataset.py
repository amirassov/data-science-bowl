import cv2
from torch.utils.data import Dataset

from dstorch.utils import to_float_tensor, pad_image


class BaseDataset(Dataset):
    def __init__(self, ids, path_images, transform):
        self.ids = ids
        self.path_images = path_images
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        raise NotImplementedError


class TrainDataset(BaseDataset):
    def __init__(self, ids, path_images, path_masks, transform):
        super().__init__(ids, path_images, transform)
        self.path_masks = path_masks

    def __getitem__(self, index):
        filename = self.ids[index]
        image = cv2.imread(self.path_images.format(filename))
        mask = cv2.imread(self.path_masks.format(filename), cv2.IMREAD_UNCHANGED)
        image, mask = self.transform(image, mask)

        return {
            'image': to_float_tensor(image),
            'mask': to_float_tensor(mask)
        }
        

class ValDataset(BaseDataset):
    def __init__(self, ids, path_images, path_masks, transform, period):
        super().__init__(ids, path_images, transform)
        self.path_masks = path_masks
        self.period = period

    def __getitem__(self, index):
        filename = self.ids[index]
        image = cv2.imread(self.path_images.format(filename))
        mask = cv2.imread(self.path_masks.format(filename), cv2.IMREAD_UNCHANGED)
        image, mask = self.transform(image, mask)

        padded_image, top, left = pad_image(image, self.period)
        padded_mask, top, left = pad_image(mask, self.period)
        height, width = image.shape[:2]
        return {
            'image': to_float_tensor(padded_image),
            'mask': to_float_tensor(padded_mask),
            'top': top,
            'left': left,
            'height': height,
            'width': width
        }

class TestDataset(BaseDataset):
    def __init__(self, ids, path_images, transform, period):
        super().__init__(ids, path_images, transform)
        self.period = period

    def __getitem__(self, index):
        filename = self.ids[index]
        image = cv2.imread(self.path_images.format(filename))
        image = self.transform(image)[0]

        padded_mask, top, left = pad_image(image, self.period)
        height, width = image.shape[:2]
        return {
            'image': to_float_tensor(pad_image),
            'top': top,
            'left': left,
            'height': height,
            'width': width
        }
