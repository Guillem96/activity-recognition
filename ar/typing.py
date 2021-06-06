from pathlib import Path
from typing import Any
from typing import Callable
from typing import Optional
from typing import Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import Subset
from torch.utils.tensorboard import SummaryWriter

Transform = Callable[[Union[torch.Tensor, Image.Image]], torch.Tensor]

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

Number = Union[int, float, torch.Tensor]
SubOrDataset = Union[Dataset, Subset]

Optimizer = Union[torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW]
Scheduler = Union[torch.optim.lr_scheduler.OneCycleLR,  # type: ignore
                  torch.optim.lr_scheduler.StepLR]

TensorBoard = Optional[SummaryWriter]

PathLike = Union[Path, str]
