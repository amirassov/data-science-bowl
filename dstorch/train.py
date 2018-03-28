import os
from collections import defaultdict
from copy import deepcopy

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter
from tqdm import tqdm

from dstorch.losses import BCEDiceLoss, BCEDiceLossCenters, DiceLoss, BCEDiceLossOneClass
from dstorch.models import TernausNet34
from dstorch.utils import variable
from dstorch.dataset import TrainDataset, ValDataset

import numpy as np


models = {
    'TernausNet34': TernausNet34
}

losses = {
    'BCEDiceLoss': BCEDiceLoss,
    'BCEDiceLossCenters': BCEDiceLossCenters
}


def adjust_lr(optimizer, epoch, init_lr=0.1, num_epochs_per_decay=10, lr_decay_factor=0.1):
    lr = init_lr * (lr_decay_factor ** (epoch // num_epochs_per_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def cyclic_lr(optimizer, epoch, init_lr=1e-4, num_epochs_per_cycle=5, cycle_epochs_decay=2, lr_decay_factor=0.5):
    epoch_in_cycle = epoch % num_epochs_per_cycle
    lr = init_lr * (lr_decay_factor ** (epoch_in_cycle // cycle_epochs_decay))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


class PytorchTrain:
    def __init__(
            self, model_name, network, nb_epoch, loss,
            lr, model_dir, log_dir, metrics, network_args, loss_args,
            cycle_start_epoch
    ):
        self.model_name = model_name
        self.nb_epoch = nb_epoch
        
        self.log_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.model_dir = os.path.join(model_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)

        self.model = models[network](**network_args)
        self.lr = lr
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.model = nn.DataParallel(self.model).cuda()

        self.criterion = losses[loss](**loss_args).cuda()
        self.writer = SummaryWriter(self.log_dir)
        self.metrics = metrics

        self.cycle_start_epoch = cycle_start_epoch
    
    def run_one_epoch(self, epoch, lr, loader, training=True):
        epoch_report = defaultdict(float)
        
        if training:
            progress_bar = tqdm(
                enumerate(loader), total=len(loader),
                desc="Epoch {}, lr {}".format(epoch, lr), ncols=0
            )
        else:
            progress_bar = enumerate(loader)
        
        for i, data in progress_bar:
            step_report = self.make_step(data, training)
            
            for key, value in step_report.items():
                epoch_report[key] += value
            
            if training:
                progress_bar.set_postfix(
                    **{key: "{:.5f}".format(value.cpu().numpy()[0] / (i + 1)) for key, value in epoch_report.items()}
                )
        
        return {key: value.cpu().numpy()[0] / len(loader) for key, value in epoch_report.items()}
    
    def make_step(self, data, training):
        return {
            True: self.make_train_step,
            False: self.make_val_step
        }[training](data)
    
    def make_train_step(self, data):
        report = {}
        images = variable(data['image'])
        masks = variable(data['mask'])
        
        self.optimizer.zero_grad()
        
        predictions = self.model(images)
        loss = self.criterion(predictions, masks)
        
        report['loss'] = loss.data
        for name, func in self.metrics:
            report[name] = func(predictions, masks).data
        
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.0)
        self.optimizer.step()
        return report
    
    def make_val_step(self, data):
        report = {}
        
        top, left, height, width = data['top'][0], data['left'][0], data['height'][0], data['width'][0]
        image = variable(data['image'], volatile=True)
        mask = variable(data['mask'], volatile=True)
        
        prediction = self.model(image)
        loss = self.criterion(prediction, mask)
        report['loss'] = loss.data
        
        for name, func in self.metrics:
            report[name] = func(
                prediction[:, :, top:height + top, left:width + left].contiguous(),
                mask[:, :, top:height + top, left:width + left].contiguous()
            ).data
        
        return report
    
    def fit(self, train_loader, val_loader):
        best_epoch = -1
        best_loss = float('inf')
        try:
            for epoch in range(self.nb_epoch):
                lr = np.repeat((1.5 ** np.arange(30)) * 1e-7, 2)[epoch]
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                    
                self.model.train()
                train_metrics = self.run_one_epoch(epoch, lr, train_loader)
                
                self.model.eval()
                val_metrics = self.run_one_epoch(epoch, lr, val_loader, training=False)
                
                print(" | ".join("{}: {:.5f}".format(key, float(value)) for key, value in val_metrics.items()))
                
                for key, value in train_metrics.items():
                    self.writer.add_scalar('train/{}'.format(key), float(value), global_step=epoch)
                
                for key, value in val_metrics.items():
                    self.writer.add_scalar('val/{}'.format(key), float(value), global_step=epoch)
                
                self.writer.add_scalar('lr', float(lr), global_step=epoch)
                
                loss = float(val_metrics['loss'])
                if loss < best_loss:
                    best_loss = loss
                    best_epoch = epoch
                    torch.save(deepcopy(self.model), os.path.join(self.model_dir, 'fold_best.pth'))
        except KeyboardInterrupt:
            print('done.')
        
        print('Finished: best loss {:.5f} on epoch {}'.format(best_loss, best_epoch))


def train(
        train_args,
        train_ids, val_ids, path_images, path_masks,
        batch_size=16, num_workers=1,
        train_transforms=None, val_transforms=None, period=64,
):
    train_dataset = TrainDataset(train_ids, path_images, path_masks, train_transforms)
    val_dataset = ValDataset(val_ids, path_images, path_masks, val_transforms, period)
    
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        dataset=val_dataset, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    trainer = PytorchTrain(
        **train_args,
        metrics=[
            ('bce', nn.modules.loss.BCEWithLogitsLoss()),
            ('dice', DiceLoss()),
            ('0 cls', BCEDiceLossOneClass(0)),
            ('1 cls', BCEDiceLossOneClass(1)),
            ('2 cls', BCEDiceLossOneClass(2)),
        ]
    )
    
    trainer.fit(train_loader, val_loader)
    trainer.writer.close()
