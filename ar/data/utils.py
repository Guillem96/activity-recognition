from typing import Any
from typing import Callable
from typing import Sequence
from typing import Tuple

import torch
from torchvision.datasets.samplers import RandomClipSampler
from torchvision.datasets.samplers import UniformClipSampler

from ar.data import ClipLevelDataset

_CollateFn = Callable[[Sequence[Any]], Tuple[torch.Tensor, ...]]


def image_default_collate_fn(
        batch: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch dataloader collate function for an image dataset.

    The dataset is expected to return a tuple of tensors with the image
    and the encoded label.

    Parameters
    ----------
    batch: Sequence[Any]
        Dataset

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple of batches containing the images and labels
    """
    images, labels = zip(*batch)
    return torch.stack(images), torch.as_tensor(labels)


def video_default_collate_fn(
        batch: Sequence[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch dataloader collate function for Video dataset.

    Parameters
    ----------
    batch: Sequence[Any]
        Samples of `ar.data.ClipLevelDataset`

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Batch of videos and labels.
    """
    video, _, label, _ = zip(*batch)
    return torch.stack(video), torch.as_tensor(label)


def batch_data(
    ds: ClipLevelDataset,
    batch_size: int,
    *,
    is_train: bool,
    workers: int = 1,
    collate_fn: _CollateFn = video_default_collate_fn
) -> torch.utils.data.DataLoader:
    if is_train:
        sampler = RandomClipSampler(ds.video_clips, 10)
    else:
        sampler = UniformClipSampler(ds.video_clips, 10)

    return torch.utils.data.DataLoader(ds,
                                       batch_size=batch_size,
                                       num_workers=workers,
                                       sampler=sampler,
                                       collate_fn=collate_fn,
                                       pin_memory=True)
