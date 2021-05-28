from typing import Collection
from typing import Mapping
from typing import Optional
from typing import Sequence

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn

import ar
from ar.utils.checkpoint import SerializableModule

from .logger import LogValue
from .logger import ValuesLogger
from .nn import get_lr

_DEFAULT_METRICS = (ar.metrics.top_3_accuracy, ar.metrics.top_5_accuracy,
                    ar.metrics.accuracy)


def seed(seed: int = 0) -> None:
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True # type: ignore
    # torch.backends.cudnn.benchmark = False # type: ignore
    np.random.seed(seed)


def train_one_epoch(
    dl: torch.utils.data.DataLoader,
    model: nn.Module,
    optimizer: ar.typing.Optimizer,
    loss_fn: ar.typing.LossFn,
    epoch: int,
    metrics: Collection[ar.typing.MetricFn] = (),
    grad_accum_steps: int = 1,
    scheduler: ar.typing.Scheduler = None,
    summary_writer: ar.typing.TensorBoard = None,
    mixed_precision: bool = False,
    device: torch.device = torch.device('cpu')
) -> None:

    metrics_log = [LogValue(m.__name__, len(dl)) for m in metrics]
    logger = ValuesLogger(LogValue('loss', window_size=len(dl)),
                          LogValue('lr', 1),
                          *metrics_log,
                          total_steps=len(dl),
                          header=f'Epoch[{epoch}]')

    model.train()
    optimizer.zero_grad()

    scaler: Optional[amp.GradScaler] = None
    if mixed_precision:
        scaler = amp.GradScaler()

    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)

        with amp.autocast(enabled=mixed_precision):
            predictions = model(x)
            loss = loss_fn(predictions, y)

        if scaler is not None:
            scaler.scale(loss / grad_accum_steps).backward()
        else:
            (loss / grad_accum_steps).backward()

        if (i + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        current_metrics = {
            m.__name__: m(predictions.float(), y).item() for m in metrics
        }
        logger(loss=loss.item(), lr=get_lr(optimizer), **current_metrics)

        # Write logs to tensorboard
        step = epoch * len(dl) + i

        if summary_writer is not None:
            summary_writer.add_scalar('learning_rate',
                                      get_lr(optimizer),
                                      global_step=step)
            summary_writer.add_scalar('train_loss',
                                      loss.item(),
                                      global_step=step)
            summary_writer.add_scalars('train_metrics',
                                       current_metrics,
                                       global_step=step)

    if summary_writer is not None:
        log_values = logger.as_dict()
        del log_values['lr']

        summary_writer.add_scalar('epoch_train_loss',
                                  log_values.pop('loss'),
                                  global_step=epoch)

        summary_writer.add_scalars('epoch_train_metrics',
                                   log_values,
                                   global_step=epoch)

    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()

    optimizer.zero_grad()


@torch.no_grad()
def evaluate(
    dl: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: ar.typing.LossFn,
    metrics: Collection[ar.typing.MetricFn],
    epoch: int,
    mixed_precision: bool = False,
    summary_writer: ar.typing.TensorBoard = None,
    device: torch.device = torch.device('cpu')
) -> Mapping[str, float]:

    metrics_log = [LogValue(m.__name__, len(dl)) for m in metrics]
    logger = ValuesLogger(*metrics_log,
                          LogValue('loss', len(dl)),
                          total_steps=len(dl),
                          header='Validation')

    model.eval()
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        with amp.autocast(enabled=mixed_precision):
            predictions = model(x)
            loss = loss_fn(predictions, y)

        updates_values = {
            m.__name__: m(predictions.float(), y).item() for m in metrics
        }
        updates_values['loss'] = loss.item()
        logger(**updates_values)

    if summary_writer is not None:
        log_values = logger.as_dict()
        summary_writer.add_scalar('validation_loss',
                                  log_values.pop('loss'),
                                  global_step=epoch)
        summary_writer.add_scalars('validation_metrics',
                                   log_values,
                                   global_step=epoch)

    return logger.as_dict()


def train(
    model: SerializableModule,
    optimizer: torch.optim.Optimizer,
    train_ds: torch.utils.data.Dataset,
    valid_ds: torch.utils.data.Dataset,
    *,
    epochs: int,
    batch_size: int,
    save_checkpoint: ar.typing.PathLike,
    dl_workers: int = 1,
    train_from: dict = {},
    grad_accum_steps: int = 1,
    fp16: bool = False,
    summary_writer: Optional[ar.typing.TensorBoard] = None,
    scheduler: Optional[ar.typing.Scheduler] = None,
    metrics: Sequence[ar.typing.MetricFn] = _DEFAULT_METRICS,
    device: torch.device = torch.device('cpu')
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

    train_dl = ar.data.batch_data(train_ds, batch_size, workers=dl_workers)
    valid_dl = ar.data.batch_data(valid_ds, batch_size, workers=dl_workers)

    for epoch in range(starting_epoch, epochs):
        train_one_epoch(dl=train_dl,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_fn=criterion_fn,
                        metrics=metrics,
                        grad_accum_steps=grad_accum_steps,
                        mixed_precision=fp16,
                        epoch=epoch,
                        summary_writer=summary_writer,
                        device=device)

        eval_metrics = evaluate(dl=valid_dl,
                                model=model,
                                metrics=metrics,
                                loss_fn=criterion_fn,
                                epoch=epoch,
                                summary_writer=summary_writer,
                                mixed_precision=fp16,
                                device=device)

        # Save the model jointly with the optimizer
        model.save(
            save_checkpoint.format(epoch=epoch, model=model.__class__.__name__),
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else {})

    return eval_metrics
