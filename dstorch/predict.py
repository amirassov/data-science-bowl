from torch.nn import functional as F
from tqdm import tqdm_notebook
from dstorch.utils import variable
from dstorch.dataset import make_loader
import torch



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

def predict(model, images, ids, transform, batch_size, flips):
    loader = make_loader(
        images, masks=None, ids=ids,
        batch_size=batch_size, transform=transform,
        shuffle=False, mode='predict'
    )
    test_predictions = []
    test_names = []

    for inputs, names, tops, lefts in tqdm_notebook(loader, desc='Predict', total=len(images)):
        inputs = variable(inputs, volatile=True)
        outputs = batch_predict(model, inputs, flips=flips)

        for i, output in enumerate(outputs[:, 0]):
            height, width = output.shape[:2]
            test_predictions.append(output[tops[i]:height-tops[i], lefts[i]:width-lefts[i]])
            test_names.append(names[i])
    return test_predictions, test_names
