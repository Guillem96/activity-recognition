import abc
from typing import Callable, Union

import torch
from torch.utils.data import Dataset, Subset

from PIL import Image

Transform = Callable[[Union[torch.Tensor, 'Image']], torch.Tensor]

LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

Number = Union[int, float, torch.Tensor]
SubOrDataset = Union[Dataset, Subset]
