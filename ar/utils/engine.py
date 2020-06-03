from typing import Collection, Optional

import torch
import torch.nn as nn

# TODO: Enable mixed precision when pytorch 1.6.0
# import torch.cuda.amp as amp 

import numpy as np

from .nn import get_lr
from ar.typing import Optimizer, LossFn, MetricFn
from .logger import LogValue, ValuesLogger

def seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True # type: ignore
    torch.backends.cudnn.benchmark = False # type: ignore
    np.random.seed(seed)

    
def train_one_epoch(dl: torch.utils.data.DataLoader,
                    model: nn.Module,
                    optimizer: Optimizer,
                    loss_fn: LossFn,
                    epoch: int,
                    grad_accum_steps: int = 1,
                    scheduler: torch.optim.lr_scheduler._LRScheduler = None,
                    print_freq: int = 10,
                    # mixed_precision: bool = False,
                    device: torch.device = torch.device('cpu')) -> None:
    
    logger = ValuesLogger(
        LogValue('loss', window_size=len(dl)),
        LogValue('lr', 1),
        print_freq=print_freq,
        header='Epoch[{epoch}] [{step}/{total}]'.format(epoch=epoch,
                                                        step='{step}',
                                                        total=len(dl)))

    model.train()
    optimizer.zero_grad()

    # scaler: Optional[amp.GradScaler] = None
    # if mixed_precision:
    #     scaler = amp.GradScaler()

    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)

        # with amp.autocast(enabled=mixed_precision):
        predictions = model(x)
        
        loss = loss_fn(predictions, y)
        loss = loss / grad_accum_steps
    
        # if scaler is not None:
        #     loss = scaler.scale(loss)
            
        loss.backward()

        if ((i + 1) % grad_accum_steps) == 0:
            
            # if scaler is not None:
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            
            optimizer.step()
            optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        logger(loss=loss.item(), lr=get_lr(optimizer))

    optimizer.step()
    optimizer.zero_grad()


@torch.no_grad()
def evaluate(dl: torch.utils.data.DataLoader,
             model: nn.Module,
             loss_fn: LossFn,
             metrics: Collection[MetricFn],
            #  mixed_precision: bool = False,
             device: torch.device = torch.device('cpu')) -> None:
    
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

        # with amp.autocast(enabled=mixed_precision):
        predictions = model(x)
        loss = loss_fn(predictions, y)

        updates_values = {m.__name__: m(predictions, y).item() for m in metrics}
        updates_values['loss'] = loss.item()
        logger(**updates_values)
