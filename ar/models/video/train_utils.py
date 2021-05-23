from pathlib import Path
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets.samplers import RandomClipSampler
from torchvision.datasets.samplers import UniformClipSampler

import ar
import ar.transforms as VT
from ar.data.datasets.base import ClipLevelDataset
from ar.typing import PathLike


def load_datasets(
    dataset_type: str,
    root: ar.typing.PathLike,
    frames_per_clip: int,
    steps_between_clips: int,
    workers: int = 1,
    frame_rate: Optional[int] = None,
    annotations_path: Optional[ar.typing.PathLike] = None,
    validation_size: float = .1,
    train_transforms: Optional[ar.typing.Transform] = None,
    valid_transforms: Optional[ar.typing.Transform] = None
) -> Tuple[ar.data.ClipLevelDataset, ar.data.ClipLevelDataset]:
    """
    Given a dataset type, performs a set of operations to generate a train
    and validation dataset
    """
    cache_file = Path.home() / '.ar' / 'datasets' / f'{dataset_type}.pt'
    cache_file.parent.mkdir(exist_ok=True, parents=True)

    train_ds: Optional[ar.data.ClipLevelDataset] = None
    valid_ds: Optional[ar.data.ClipLevelDataset] = None

    if cache_file.exists():
        train_ds, valid_ds = torch.load(cache_file)
    elif dataset_type == 'kinetics400':
        train_ds = ar.data.Kinetics400(root=root,
                                       split='train',
                                       frames_per_clip=frames_per_clip,
                                       frame_rate=frame_rate,
                                       step_between_clips=steps_between_clips,
                                       extensions=('.mp4',),
                                       num_workers=workers,
                                       transform=train_transforms)

        valid_ds = ar.data.Kinetics400(root=root,
                                       split='validate',
                                       frames_per_clip=frames_per_clip,
                                       frame_rate=frame_rate,
                                       step_between_clips=steps_between_clips,
                                       extensions=('.mp4',),
                                       num_workers=workers,
                                       transform=valid_transforms)

        torch.save((train_ds, valid_ds), cache_file)
    else:  # dataset_type == 'UCF-101':
        if annotations_path is None:
            raise ValueError(f'The annotations must be provided when using '
                             f'{dataset_type}')

        train_ds = ar.data.UCF101(root=root,
                                  annotation_path=annotations_path,
                                  frames_per_clip=frames_per_clip,
                                  frame_rate=frame_rate,
                                  split='train',
                                  step_between_clips=steps_between_clips,
                                  transform=train_transforms,
                                  num_workers=workers)

        valid_ds = ar.data.UCF101(root=root,
                                  annotation_path=annotations_path,
                                  frames_per_clip=frames_per_clip,
                                  frame_rate=frame_rate,
                                  split='test',
                                  step_between_clips=steps_between_clips,
                                  transform=valid_transforms,
                                  num_workers=workers)

        torch.save((train_ds, valid_ds), cache_file)

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


def dl_samplers(
        train_ds: ClipLevelDataset, valid_ds: ClipLevelDataset
) -> Tuple[RandomClipSampler, UniformClipSampler]:
    if all(hasattr(o, 'video_clips') for o in [train_ds, valid_ds]):
        train_sampler = RandomClipSampler(train_ds.video_clips, 10)
        valid_sampler = UniformClipSampler(valid_ds.video_clips, 10)
    else:
        raise ValueError('Video dataset must have the video_clips attribute')

    return train_sampler, valid_sampler


def data_preparation(
        dataset: str,
        *,
        data_dir: PathLike,
        frames_per_clip: int,
        batch_size: int,
        video_size: Optional[Tuple[int, int]] = None,
        frame_rate: Optional[int] = None,
        size_before_crop: Optional[Tuple[int, int]] = None,
        annotations_path: Optional[PathLike] = None,
        writer: Optional[ar.typing.TensorBoard] = None,
        steps_between_clips: int = 1,
        workers: int = 1,
        validation_size: float = .1) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Loads the datasets with corresponding transformations and creates two data
    loaders, one for train and validation
    """

    size_before_crop = size_before_crop or (128, 171)
    video_size = video_size or (112, 112)

    if any(size_before_crop[i] < video_size[i] for i in range(2)):
        raise ValueError(f'size_before_crop must {size_before_crop} '
                         f'be larger than the video_size {video_size}')

    train_tfms = T.Compose([
        VT.VideoToTensor(),
        VT.VideoResize(size_before_crop),
        VT.VideoRandomCrop(video_size),
        VT.VideoRandomHorizontalFlip(),
        # VT.VideoRandomErase(scale=(0.02, 0.15))
        VT.VideoNormalize(**VT.imagenet_stats),
    ])

    valid_tfms = T.Compose([
        VT.VideoToTensor(),
        VT.VideoResize(size_before_crop),
        VT.VideoCenterCrop(video_size),
        VT.VideoNormalize(**VT.imagenet_stats),
    ])

    train_ds, valid_ds = load_datasets(dataset_type=dataset,
                                       root=data_dir,
                                       annotations_path=annotations_path,
                                       frames_per_clip=frames_per_clip,
                                       frame_rate=frame_rate,
                                       steps_between_clips=steps_between_clips,
                                       workers=workers,
                                       validation_size=validation_size,
                                       train_transforms=train_tfms,
                                       valid_transforms=valid_tfms)

    if writer:
        ar.logger.log_random_videos(train_ds,
                                    writer=writer,
                                    unnormalize_videos=True,
                                    video_format='CTHW')

    train_sampler, valid_sampler = dl_samplers(train_ds, valid_ds)

    train_dl = data.DataLoader(train_ds,
                               batch_size=batch_size,
                               num_workers=workers,
                               sampler=train_sampler,
                               collate_fn=default_collate_fn,
                               pin_memory=True)

    valid_dl = data.DataLoader(valid_ds,
                               batch_size=batch_size,
                               num_workers=workers,
                               sampler=valid_sampler,
                               collate_fn=default_collate_fn,
                               pin_memory=True)

    return train_dl, valid_dl
