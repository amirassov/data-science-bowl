from torch.nn import functional as F
from tqdm import tqdm
from dstorch.utils import variable
from dstorch.dataset import make_loader


def predict(model, images, ids, transform, batch_size):
    loader = make_loader(images, masks=None, ids=ids,
                         batch_size=batch_size, transform=transform, shuffle=False, mode='predict')
    test_predictions = []
    test_names = []

    for batch_num, (inputs, names, tops, lefts) in enumerate(tqdm(loader, desc='Predict')):
        inputs = variable(inputs, volatile=True)
        outputs = model(inputs)

        for i, outputs in enumerate(outputs):
            t_mask = (F.sigmoid(outputs[i]).data.cpu().numpy())[tops[i]:-tops[i], lefts[i]:-lefts[i]]
            test_predictions.append(t_mask)
            test_names.append(names[i])
    return test_predictions, test_names
