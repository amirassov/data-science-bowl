from torch.nn import functional as F
from tqdm import tqdm
from dstorch.utils import variable
import torch
import numpy as np
from dstorch.dataset import TestDataset
from torch.utils.data import DataLoader


def flip_tensor_lr(batch):
    invert_indices = torch.arange(batch.data.size()[-1] - 1, -1, -1).long()
    return batch.index_select(3, variable(invert_indices))

def flip_tensor_ud(batch):
    invert_indices = torch.arange(batch.data.size()[-2] - 1, -1, -1).long()
    return torch.index_select(batch, 2, variable(invert_indices))

def to_numpy(batch):
    if isinstance(batch, tuple):
        batch = batch[0]
    return F.sigmoid(batch).data.cpu().numpy()

def batch_predict(model, batch, flips=0):
    pred1 = model(batch)
    if flips > 0:
        pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
        masks = [pred1, pred2]
        if flips > 1:
            pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
            pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))
            masks.extend([pred3, pred4])
        new_mask = torch.stack(masks).mean(0)
        return to_numpy(new_mask)
    return to_numpy(pred1)

def predict(model, ids,  path_images, transforms, period, flips, num_workers):
    dataset = TestDataset(ids, path_images, transforms, period)

    loader = DataLoader(
        dataset=dataset, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    test_predictions = []

    for data in tqdm(loader, desc='Predict', total=len(ids)):
        images, tops, lefts, heights, widths = data['image'], data['top'], data['left'], data['height'], data['width']
        inputs = variable(images, volatile=True)
        predictions = batch_predict(model, inputs, flips=flips)

        for i, prediction in enumerate(predictions):
            prediction = np.moveaxis(prediction, 0, -1)
            test_predictions.append(prediction[tops[i]:heights[i] + tops[i], lefts[i]:widths[i] + lefts[i]])
    return test_predictions
