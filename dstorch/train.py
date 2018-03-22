import random

import numpy as np
from torch import nn
from tqdm import tqdm_notebook

from dstorch.utils import variable


def validation_binary(model: nn.Module, criterion, val_loader):
    model.eval()
    losses = []

    for inputs, targets, tops, lefts in val_loader:
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs = model(inputs)
        for i, output in enumerate(outputs):
            _, _, height, width = output.shape
            loss = criterion(output[tops[i]:height - tops[i], lefts[i]:width - lefts[i]], targets[i])
            losses.append(loss.data[0])

    valid_loss = np.mean(losses)

    print('Valid loss: {:.5f}'.format(valid_loss))
    metrics = {'valid_loss': valid_loss}
    return metrics


def train(model, n_epochs, batch_size, criterion, train_loader, val_loader, init_optimizer, lr):
    optimizer = init_optimizer(lr)
    epoch, report_each, valid_losses = 1, 10, []

    for epoch in range(epoch, n_epochs + 1):
        model.train()
        random.seed()
        bar = tqdm_notebook(total=(len(train_loader) * batch_size))
        bar.set_description('Epoch {}, lr {}'.format(epoch, lr))
        losses = []
        _train_loader = train_loader
        try:
            for i, (inputs, targets) in enumerate(_train_loader):
                inputs, targets = variable(inputs), variable(targets)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                batch_size = inputs.size(0)
                loss.backward()
                optimizer.step()
                
                bar.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                bar.set_postfix(loss='{:.5f}'.format(mean_loss))

            bar.close()
            valid_metrics = validation_binary(model, criterion, val_loader)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        
        except KeyboardInterrupt:
            bar.close()
            print('done.')
            return
