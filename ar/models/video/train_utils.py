from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.optim as optim

import ar


def load_datasets(
    dataset_type: str,
    root: ar.typing.PathLike,
    frames_per_clip: int,
    steps_between_clips: int,
    workers: int = 1,
    annotations_path: Optional[ar.typing.PathLike] = None,
    validation_size: float = .1,
    train_transforms: ar.typing.Transform = None,
    valid_transforms: ar.typing.Transform = None
) -> Tuple[ar.data.ClipLevelDataset, ar.data.ClipLevelDataset]:
    """
    Given a dataset type, performs a set of operations to generate a train
    and validation dataset
    """

    train_ds: Optional[ar.data.ClipLevelDataset] = None
    valid_ds: Optional[ar.data.ClipLevelDataset] = None
    if dataset_type == 'kinetics400':
        train_ds = ar.data.Kinetics400(root=root,
                                       split='train',
                                       frames_per_clip=frames_per_clip,
                                       step_between_clips=steps_between_clips,
                                       extensions=('.mp4',),
                                       num_workers=workers,
                                       transform=train_transforms)

        valid_ds = ar.data.Kinetics400(root=root,
                                       split='validate',
                                       frames_per_clip=frames_per_clip,
                                       step_between_clips=steps_between_clips,
                                       extensions=('.mp4',),
                                       num_workers=workers,
                                       transform=valid_transforms)

    else:  # dataset_type == 'UCF-101':
        if annotations_path is None:
            raise ValueError(f'The annotations must be provided when using '
                             f'{dataset_type}')

        train_ds = ar.data.UCF101(root=root,
                                  annotation_path=annotations_path,
                                  frames_per_clip=frames_per_clip,
                                  split='train',
                                  step_between_clips=steps_between_clips,
                                  transform=train_transforms,
                                  num_workers=workers)

        valid_ds = ar.data.UCF101(root=root,
                                  annotation_path=annotations_path,
                                  frames_per_clip=frames_per_clip,
                                  split='test',
                                  step_between_clips=steps_between_clips,
                                  transform=valid_transforms,
                                  num_workers=workers)

    return train_ds, valid_ds


def load_optimizer(
    model: torch.nn.Module,
    optimizer_type: str,
    scheduler_type: str,
    *,
    checkpoint: dict,
    learning_rate: float,
    epochs: int = -1,
    steps_per_epoch: int = -1
) -> Tuple[ar.typing.Optimizer, Optional[ar.typing.Scheduler]]:
    """
    Load an optimizer to update the model parameters and if specified,
    also creates a learning rate scheduler.

    If a checkpoint dict is provided, this loads the optimizer and scheduler 
    state dicts
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer: Optimizer = None  # type: ignore
    if optimizer_type == 'AdamW':
        optimizer = optim.AdamW(trainable_params, lr=learning_rate)
    elif optimizer_type == 'SGD':
        optimizer = optim.SGD(trainable_params,
                              lr=learning_rate,
                              momentum=0.9,
                              weight_decay=4e-5)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(trainable_params, lr=learning_rate)

    if scheduler_type == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(  # type: ignore
            optimizer, learning_rate * 10, steps_per_epoch * epochs)
    elif scheduler_type == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, steps_per_epoch, .1)
    elif scheduler_type == 'None':
        scheduler = None

    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if 'scheduler' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return optimizer, scheduler


def default_collate_fn(
        batch: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    video, _, label, _ = zip(*batch)
    return torch.stack(video), torch.as_tensor(label)