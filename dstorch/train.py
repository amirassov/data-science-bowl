import random

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm_notebook
from torch import nn


def variable(x, volatile=False):
    if isinstance(x, (list, tuple)):
        return [variable(y, volatile=volatile) for y in x]
    return cuda(Variable(x, volatile=volatile))


def cuda(x):
    return x.cuda(async=True) if torch.cuda.is_available() else x


def validation_binary(model: nn.Module, criterion, valid_loader):
    model.eval()
    losses = []
    
    for inputs, targets in valid_loader:
        inputs = variable(inputs, volatile=True)
        targets = variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        losses.append(loss.data[0])
    
    valid_loss = np.mean(losses)

    print('Valid loss: {:.5f}'.format(valid_loss))
    metrics = {'valid_loss': valid_loss}
    return metrics


def train(model, n_epochs, batch_size, criterion, train_loader, valid_loader, init_optimizer, lr):
    optimizer = init_optimizer(lr)
    epoch, step, report_each, valid_losses = 1, 0, 10, []
    
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
                step += 1
                
                bar.update(batch_size)
                losses.append(loss.data[0])
                mean_loss = np.mean(losses[-report_each:])
                bar.set_postfix(loss='{:.5f}'.format(mean_loss))

            bar.close()
            valid_metrics = validation_binary(model, criterion, valid_loader)
            valid_loss = valid_metrics['valid_loss']
            valid_losses.append(valid_loss)
        
        except KeyboardInterrupt:
            bar.close()
            print('done.')
            return
