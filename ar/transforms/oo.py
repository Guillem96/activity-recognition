import math
import random
from typing import Callable, Collection, Tuple, Sequence, List, Union

import torch
from . import functional as F

Transform = Callable[[torch.Tensor], torch.Tensor]


class VideoRandomCrop(torch.nn.Module):
    def __init__(self, size: Tuple[int, int]) -> None:
        super(VideoRandomCrop, self).__init__()
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

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        i, j, h, w = self.get_params(vid, self.size)
        return F.video_crop(vid, i, j, h, w)


class VideoRandomErase(torch.nn.Module):
    def __init__(self,
                 p: float = 0.5,
                 scale: Tuple[float, float] = (0.02, 0.33),
                 ratio: Tuple[float, float] = (0.3, 3.3),
                 value: float = 0.) -> None:
        super(VideoRandomErase, self).__init__()
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            raise ValueError('range should be of kind (min, max)')
        if scale[0] < 0 or scale[1] > 1:
            raise ValueError('range of scale should be between 0 and 1')
        if p < 0 or p > 1:
            raise ValueError('range of random erasing probability'
                             'should be between 0 and 1')

        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    @staticmethod
    def get_params(
        video: torch.Tensor,
        scale: Tuple[float, float],
        ratio: Tuple[float, float],
        value: float = 0.
    ) -> Tuple[int, int, int, int, Union[float, torch.Tensor]]:
        """Get parameters for ``video_erase`` for a random erasing.

        Args:
        video: torch.Tensor 
            Tensor image of size (C, FRAMES, H, W) to be erased.
        scale: Tuple[float, float]
            Range of proportion of erased area against input image.
        ratio: Tuple[float, float]
            range of aspect ratio of erased area.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        c, f, h, w = video.shape
        area = h * w

        for _ in range(10):
            erase_area = random.uniform(scale[0], scale[1]) * area
            aspect_ratio = random.uniform(ratio[0], ratio[1])

            e_h = int(round(math.sqrt(erase_area * aspect_ratio)))
            e_w = int(round(math.sqrt(erase_area / aspect_ratio)))

            if e_h < h and e_w < w:
                i = random.randint(0, h - e_h)
                j = random.randint(0, w - e_w)

                return i, j, e_h, e_w, value

        # Return original image
        return 0, 0, h, w, video

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        video: torch.Tensor 
            Tensor image of size (C, FRAMES, H, W) to be erased.

        Returns
        -------
        torch.Tensor
        """
        if random.random() < self.p:
            x, y, h, w, v = self.get_params(video,
                                            scale=self.scale,
                                            ratio=self.ratio,
                                            value=self.value)

            return F.video_erase(video, x, y, h, w, v)

        return video


class VideoCenterCrop(torch.nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super(VideoCenterCrop, self).__init__()
        self.size = size

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return F.video_center_crop(vid, self.size)


class VideoResize(torch.nn.Module):
    def __init__(self, size: Tuple[int, int]):
        super(VideoResize, self).__init__()
        self.size = size

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return F.video_resize(vid, self.size)


class VideoToTensor(torch.nn.Module):
    def __init__(self):
        super(VideoToTensor, self).__init__()

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return F.video_to_tensor(vid)


class VideoNormalize(torch.nn.Module):
    def __init__(self, mean: Tuple[float, float, float],
                 std: Tuple[float, float, float]):
        super(VideoNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return F.video_normalize(vid, self.mean, self.std)


class VideoRandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super(VideoRandomHorizontalFlip, self).__init__()
        self.p = p

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        if random.random() < self.p:
            return F.video_hflip(vid)
        return vid


class VideoPad(torch.nn.Module):
    def __init__(self, padding: List[int], fill: int = 0):
        super(VideoPad, self).__init__()
        self.padding = padding
        self.fill = fill

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        return F.video_pad(vid, self.padding, self.fill)


class OneOf(torch.nn.Module):
    def __init__(self, transforms: Sequence[Transform]) -> None:
        super(OneOf, self).__init__()

        self.transforms = transforms

    def forward(self, vid: torch.Tensor) -> torch.Tensor:
        tfm_fn: Transform = random.choice(self.transforms)
        return tfm_fn(vid)


# Does nothing tho the input tensor
Identity = lambda vid: vid
