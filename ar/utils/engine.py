from typing import Callable, Collection

import torch
import torch.nn as nn

import numpy as np

from .nn import get_lr
from .logger import LogValue, ValuesLogger


LossFn = Callable[[torch.FloatTensor, torch.LongTensor], torch.FloatTensor]
MetricFn = Callable[[torch.FloatTensor, torch.LongTensor], torch.FloatTensor]


def seed(seed: int = 0):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    
def train_one_epoch(dl: torch.utils.data.DataLoader,
                    model: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    loss_fn: LossFn,
                    epoch: int,
                    grad_accum_steps: int = 1,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    print_freq: int = 10,
                    device: torch.device = torch.device('cpu')):
    
    logger = ValuesLogger(
        LogValue('loss', window_size=len(dl)),
        LogValue('lr', 1),
        print_freq=print_freq,
        header='Epoch[{epoch}] [{step}/{total}]'.format(epoch=epoch,
                                                        step='{step}',
                                                        total=len(dl)))

    model.train()
    optimizer.zero_grad()

    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)

        predictions = model(x)
        
        loss = loss_fn(predictions, y)
        loss = loss / grad_accum_steps
        loss.backward()

        if ((i + 1) % grad_accum_steps) == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        logger(loss=loss.item(), lr=get_lr(optimizer))


@torch.no_grad()
def evaluate(dl: torch.utils.data.DataLoader,
             model: nn.Module,
             loss_fn: LossFn,
             metrics: Collection[MetricFn],
             device: torch.device = torch.device('cpu')):
    
    metrics_log = [LogValue(m.__name__, len(dl)) for m in metrics]
    logger = ValuesLogger(
        *metrics_log,
        LogValue('loss', len(dl)),
        print_freq=len(dl),
        header='Validation')
    
    model.eval()
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        predictions = model(x)

        loss = loss_fn(predictions, y)
        updates_values = {m.__name__: m(predictions, y) for m in metrics}
        updates_values['loss'] = loss.item()
        logger(**updates_values)

