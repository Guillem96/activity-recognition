from typing import Collection, Optional, Mapping

import torch
import torch.nn as nn

# TODO: Enable mixed precision when pytorch 1.6.0
# import torch.cuda.amp as amp 

import numpy as np

from .nn import get_lr
from .logger import LogValue, ValuesLogger
from ar.typing import (Optimizer, LossFn, MetricFn, TensorBoard, Scheduler, 
                       TensorBoard)


def seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True # type: ignore
    # torch.backends.cudnn.benchmark = False # type: ignore
    np.random.seed(seed)


def train_one_epoch(
        dl: torch.utils.data.DataLoader,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: LossFn,
        epoch: int,
        grad_accum_steps: int = 1,
        scheduler: Scheduler = None,
        summary_writer: TensorBoard = None,
        # mixed_precision: bool = False,
        device: torch.device = torch.device('cpu')) -> None:
    
    logger = ValuesLogger(
        LogValue('loss', window_size=len(dl)),
        LogValue('lr', 1),
        total_steps=len(dl),
        header=f'Epoch[{epoch}]')

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
    
        # if scaler is not None:
        #     loss = scaler.scale(loss)
            
        (loss / grad_accum_steps).backward()

        if (i + 1) % grad_accum_steps == 0:
            
            # if scaler is not None:
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            
            optimizer.step()
            optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        logger(loss=loss.item(), lr=get_lr(optimizer))
        
        # Write logs to tensorboard
        step = epoch * len(dl) + i

        if summary_writer is not None:
            summary_writer.add_scalar('learning_rate', 
                                    get_lr(optimizer), 
                                    global_step=step)
            summary_writer.add_scalars('train', {'loss': loss.item()},
                                    global_step=step)

    optimizer.step()
    optimizer.zero_grad()


@torch.no_grad()
def evaluate(dl: torch.utils.data.DataLoader,
             model: nn.Module,
             loss_fn: LossFn,
             metrics: Collection[MetricFn],
            #  mixed_precision: bool = False,
             epoch: int,
             summary_writer: TensorBoard = None,
             device: torch.device = torch.device('cpu')) -> Mapping[str, float]:
    
    metrics_log = [LogValue(m.__name__, len(dl)) for m in metrics]
    logger = ValuesLogger(
        *metrics_log,
        LogValue('loss', len(dl)),
        total_steps=len(dl),
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

    if summary_writer is not None:
        summary_writer.add_scalars('validation', logger.as_dict(),
                                   global_step=epoch)

    return logger.as_dict()
