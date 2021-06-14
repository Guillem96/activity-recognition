import random
from typing import Collection
from typing import Mapping
from typing import Optional
from typing import Sequence

import accelerate
import numpy as np
import torch
import torch.nn as nn

import ar
from ar.utils.checkpoint import SerializableModule

from .logger import LogValue
from .logger import ValuesLogger
from .nn import get_lr

_DEFAULT_METRICS = (ar.metrics.top_3_accuracy, ar.metrics.top_5_accuracy,
                    ar.metrics.accuracy)


def seed(seed: int = 0) -> None:
    """Sets random seed for reproducibility

    Parameters
    ----------
    seed : int, defaults 0
        Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    dl: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: ar.typing.Optimizer,
    loss_fn: ar.typing.LossFn,
    epoch: int,
    accelerator: accelerate.Accelerator,
    grad_accum_steps: int = 1,
    scheduler: ar.typing.Scheduler = None,
    summary_writer: ar.typing.TensorBoard = None,
) -> None:
    logger = ValuesLogger(LogValue('loss', window_size=len(dl)),
                          LogValue('lr', 1),
                          total_steps=len(dl),
                          header=f'Epoch[{epoch}]',
                          disable=not accelerator.is_local_main_process)

    model.train()
    optimizer.zero_grad()

    for i, (x, y) in enumerate(dl):
        predictions = model(x)
        loss = loss_fn(predictions, y)
        accelerator.backward(loss / grad_accum_steps)

        if i % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        logger(loss=loss.item(), lr=get_lr(optimizer))

        # Write logs to tensorboard
        step = epoch * len(dl) + i

        if accelerator.is_main_process and summary_writer:
            summary_writer.add_scalar('learning_rate',
                                      get_lr(optimizer),
                                      global_step=step)
            summary_writer.add_scalar('train_loss',
                                      loss.item(),
                                      global_step=step)

    if summary_writer and accelerator.is_main_process:
        log_values = logger.as_dict()
        del log_values['lr']

        summary_writer.add_scalar('epoch_train_loss',
                                  log_values.pop('loss'),
                                  global_step=epoch)

    logger.close()


@torch.no_grad()
def evaluate(
    dl: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: ar.typing.LossFn,
    metrics: Collection[ar.typing.MetricFn],
    epoch: int,
    accelerator: accelerate.Accelerator,
    summary_writer: ar.typing.TensorBoard = None,
) -> Mapping[str, float]:

    metrics_log = [LogValue(m.__name__, len(dl)) for m in metrics]
    logger = ValuesLogger(*metrics_log,
                          LogValue('loss', len(dl)),
                          total_steps=len(dl),
                          header='Validation',
                          disable=not accelerator.is_local_main_process)

    model.eval()
    for x, y in dl:
        predictions = model(x)
        loss = loss_fn(predictions, y)

        predictions = accelerator.gather(predictions)
        y = accelerator.gather(y)

        updates_values = {m.__name__: m(predictions, y).item() for m in metrics}
        updates_values['loss'] = loss.item()
        logger(**updates_values)

    if summary_writer and accelerator.is_main_process:
        log_values = logger.as_dict()
        summary_writer.add_scalar('validation_loss',
                                  log_values.pop('loss'),
                                  global_step=epoch)
        summary_writer.add_scalars('validation_metrics',
                                   log_values,
                                   global_step=epoch)

    logger.close()
    return logger.as_dict()


def train(
    model: SerializableModule,
    optimizer: torch.optim.Optimizer,
    train_ds: ar.data.ClipLevelDataset,
    valid_ds: ar.data.ClipLevelDataset,
    accelerator: accelerate.Accelerator,
    *,
    epochs: int,
    batch_size: int,
    save_checkpoint: ar.typing.PathLike,
    dl_workers: int = 1,
    train_from: dict = {},
    grad_accum_steps: int = 1,
    summary_writer: Optional[ar.typing.TensorBoard] = None,
    scheduler: Optional[ar.typing.Scheduler] = None,
    metrics: Sequence[ar.typing.MetricFn] = _DEFAULT_METRICS,
) -> Mapping[str, float]:
    """Train a model

    Parameters
    ----------
    model: SerializableModule
        Model to train
    optimizer: torch.optim.Optimizer
        Optimizer instance to train the model.
    train_ds: torch.utils.data.Dataset
        Dataset containing the training examples
    valid_ds: torch.utils.data.Dataset
        Dataset containing the validation examples.
    accelerator: accelerate.Accelerator
        Device acceleration specs
    epochs: int
        Number of iterations over all the dataset
    batch_size: int
        Load dataset samples in groups of `batch_size` examples.
    save_checkpoint: ar.typing.PathLike
        Path to serialize the model. Allows the {epoch} and {model} 
        placeholders.
    dl_workers: int, defaults 1
        Threads to generate the batches.
    train_from: dict, defaults {}
        Checkpoint to resume the training.
    grad_accum_steps: int, default 1
        Gradient accumulation steps.
    fp16: bool, defaults False
        Train with mixed precision?
    summary_writer: Optional[ar.typing.TensorBoard]
        Tensorboard summary writter.
    scheduler: Optional[ar.typing.Scheduler]
        Learning rate scheduler
    metrics: Sequence[ar.typing.MetricFn]
        Metrics to evaluate
    device: torch.device, default torch.device('cpu')
        Device where to run the training.

    Returns
    -------
    Mapping[str, float]
        Metrics of the validation set
    """
    criterion_fn = torch.nn.NLLLoss()
    starting_epoch = train_from.get('epoch', -1) + 1

    train_dl = ar.data.batch_data(train_ds,
                                  batch_size,
                                  workers=dl_workers,
                                  is_train=True)

    valid_dl = ar.data.batch_data(valid_ds,
                                  batch_size,
                                  workers=dl_workers,
                                  is_train=False)

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl)

    for epoch in range(starting_epoch, epochs):
        train_one_epoch(dl=train_dl,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=criterion_fn,
                        grad_accum_steps=grad_accum_steps,
                        epoch=epoch,
                        summary_writer=summary_writer,
                        accelerator=accelerator)

        eval_metrics = evaluate(dl=valid_dl,
                                model=model,
                                metrics=metrics,
                                loss_fn=criterion_fn,
                                epoch=epoch,
                                summary_writer=summary_writer,
                                accelerator=accelerator)

        # Save the model jointly with the optimizer
        model.save(
            str(save_checkpoint).format(epoch=epoch,
                                        model=model.__class__.__name__),
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else {})

    return eval_metrics
