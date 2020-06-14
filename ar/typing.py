import abc
from typing import Callable, Union

import torch
from torch.utils.data import Dataset, Subset
from torch.utils.tensorboard import SummaryWriter

from PIL import Image

# from ar.utils.logger import DummySummaryWritter


Transform = Callable[[Union[torch.Tensor, 'Image']], torch.Tensor]

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

Number = Union[int, float, torch.Tensor]
SubOrDataset = Union[Dataset, Subset]

Optimizer = Union[torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW]
Scheduler = Union[torch.optim.lr_scheduler.OneCycleLR, # type: ignore
                  torch.optim.lr_scheduler.StepLR]

TensorBoard = Union[SummaryWriter, 'DummySummaryWritter']
