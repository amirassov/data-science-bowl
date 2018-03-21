from torch.nn import functional as F
from tqdm import tqdm
from dstorch.utils import variable
from dstorch.dataset import make_loader
import torch


class flip:
    FLIP_NONE = 0
    FLIP_LR = 1
    FLIP_FULL = 2


def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]
    return torch.index_select(batch, 3, variable(torch.LongTensor(list(reversed(range(columns))))))

def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    return torch.index_select(batch, 2, variable(torch.LongTensor(list(reversed(range(rows))))))

def to_numpy(batch):
    if isinstance(batch, tuple):
        batch = batch[0]
    return F.sigmoid(batch).data.cpu().numpy()

def batch_predict(model, batch, flips=flip.FLIP_NONE):
    pred1 = model(batch)
    if flips > flip.FLIP_NONE:
        pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))
        masks = [pred1, pred2]
        if flips > flip.FLIP_LR:
            pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
            pred4 = flip_tensor_ud(flip_tensor_lr(model(flip_tensor_ud(flip_tensor_lr(batch)))))
            masks.extend([pred3, pred4])
        new_mask = torch.mean(torch.stack(masks), dim=0)
        return to_numpy(new_mask)
    return to_numpy(pred1)


def predict(model, images, ids, transform, batch_size):
    loader = make_loader(images, masks=None, ids=ids,
                         batch_size=batch_size, transform=transform, shuffle=False, mode='predict')
    test_predictions = []
    test_names = []

    for batch_num, (inputs, names, tops, lefts) in enumerate(tqdm(loader, desc='Predict')):
        inputs = variable(inputs, volatile=True)
        outputs = batch_predict(model, inputs, flips=flip.FLIP_FULL)

        for i, output in enumerate(outputs):
            print(output.shape)
            height, width = output.shape[:2]
            prediction = output[tops[i]:height-tops[i], lefts[i]:width-lefts[i]]
            test_predictions.append(prediction)
            test_names.append(names[i])
    return test_predictions, test_names
