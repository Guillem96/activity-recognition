from typing import Callable, Collection, Tuple

import torch
import ar.transforms.functional as F

Transform = Callable[[torch.Tensor], torch.Tensor]


class RandomCrop(object):

    def __init__(self, size):
        self.size = size

    @staticmethod
    def get_params(vid: torch.Tensor, 
                   output_size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random crop.
        """
        h, w = vid.shape[-2:]
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        i, j, h, w = self.get_params(vid, self.size)
        return F.crop(vid, i, j, h, w)


class CenterCrop(object):

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return F.center_crop(vid, self.size)


class Resize(object):

    def __init__(self, size: Tuple[int, int]):
        self.size = size

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return F.resize(vid, self.size)


class ToFloatTensorInZeroOne(object):

    def __call__(self, vid: torch.Tensor) -> torch.FloatTensor:
        return F.to_normalized_float_tensor(vid)


class Normalize(object):
    def __init__(self, 
                 mean: Tuple[float, float, float], 
                 std: Tuple[float, float, float]):
        self.mean = mean
        self.std = std

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return F.normalize(vid, self.mean, self.std)


class RandomHorizontalFlip(object):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return F.hflip(vid)
        return vid


class Pad(object):
    def __init__(self, padding: int, fill: int = 0):
        self.padding = padding
        self.fill = fill

    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        return F.pad(vid, self.padding, self.fill)


class OneOf(object):
    
    def __init__(self, transforms: Collection[Transform]):
        self.transforms = transforms
    
    def __call__(self, vid: torch.Tensor) -> torch.Tensor:
        tfm_fn = random.choice(self.transforms)
        return tfm_fn(vid)


# Does nothing tho the input tensor
Identity = lambda vid: vid
