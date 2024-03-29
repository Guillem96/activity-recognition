"""
Code adapted from 
https://github.com/pytorch/vision/blob/master/references/video_classification
"""

from typing import List
from typing import Tuple
from typing import Union

import torch


def video_crop(vid: torch.Tensor, i: int, j: int, h: int,
               w: int) -> torch.Tensor:
    """Crops a video 

    Extract a crop from a video given a coordinates expressed in (i, j, h, w).
    (i, j) point corresponds to the top left position of the crop
    
    Parameters
    ----------
    vid: torch.Tensor
        The video that is going to be cropped out
    i: int
        First Top left coordinate component of the crop. 
        Expressed in absolute scale
    j: int
        Second top left coordinate component of the crop. 
        Expressed in absolute scale
    h: int
        The crop height
    w: int 
        The crop width
    
    Returns
    -------
    torch.Tensor
    """
    return vid[..., i:(i + h), j:(j + w)]


def video_center_crop(vid: torch.Tensor,
                      output_size: Tuple[int, int]) -> torch.Tensor:
    """
    Calls the function `crop` with i an j being the center of the video.

    Parameters
    ----------
    vid: torch.Tensor
        Video that is going to be cropped
    output_size: Tuple[int, int]
        Crop size. (height, width)
    
    Returns
    -------
    torch.Tensor
    """
    h, w = vid.shape[-2:]
    th, tw = output_size

    i = int(round((h - th) / 2.))
    j = int(round((w - tw) / 2.))
    return video_crop(vid, i, j, th, tw)


def video_erase(video: torch.Tensor,
                i: int,
                j: int,
                h: int,
                w: int,
                v: Union[float, torch.Tensor] = 0.) -> torch.Tensor:
    """ Erase the input Tensor Image with given value.

    Parameters
    ----------
    video: torch.Tensor 
        Tensor video of size (C, FRAMES, H, W) to be erased
    i: int 
        i in (i, j) i.e coordinates of the upper left corner.
    j: int 
        j in (i,j) i.e coordinates of the upper left corner.
    h: int 
        Height of the erased region.
    w: int 
        Width of the erased region.
    v: float, default 0.
        Erasing value.

    Returns
    -------
    torch.Tensor: Erased video.
    """
    video[..., i:i + h, j:j + w] = v
    return video


def video_hflip(vid: torch.Tensor) -> torch.Tensor:
    return vid.flip(dims=[-1])


# NOTE: for those functions, which generally expect mini-batches, we keep them
# as non-minibatch so that they are applied as if they were 4d (thus image).
# this way, we only apply the transformation in the spatial domain
def video_resize(vid: torch.Tensor,
                 size: Tuple[int, int],
                 interpolation: str = 'bilinear') -> torch.Tensor:

    return torch.nn.functional.interpolate(vid,
                                           size=size,
                                           scale_factor=None,
                                           mode=interpolation,
                                           align_corners=False)


def video_to_tensor(vid: torch.Tensor) -> torch.Tensor:
    """Converts a stack of frames to a video tensor tensor.

    Parameters
    ----------
    vid: torch.Tensor of shape (FRAMES, HEIGHT, WIDTH, CHANNELS)
        Given stack of frames

    Returns
    -------
    torch.Tensor of shape (CHANNELS, FRAMES, HEIGHT, WIDTH)
    """
    return vid.permute(3, 0, 1, 2).to(torch.float32) / 255.


def video_normalize(vid: torch.Tensor, mean: Tuple[float, float, float],
                    std: Tuple[float, float, float]) -> torch.Tensor:
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean_ = torch.as_tensor(mean).reshape(shape)
    std_ = torch.as_tensor(std).reshape(shape)
    return (vid - mean_) / std_


def video_unnormalize(vid: torch.Tensor, mean: Tuple[float, float, float],
                      std: Tuple[float, float, float]) -> torch.Tensor:
    shape = (-1,) + (1,) * (vid.dim() - 1)
    mean_ = torch.as_tensor(mean).reshape(shape)
    std_ = torch.as_tensor(std).reshape(shape)
    return vid.mul(std_).add(mean_)
