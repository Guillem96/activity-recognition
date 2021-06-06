from .factory import train_tfms
from .factory import valid_tfms
from .oo import Identity
from .oo import OneOf
from .oo import VideoCenterCrop
from .oo import VideoNormalize
from .oo import VideoRandomCrop
from .oo import VideoRandomErase
from .oo import VideoRandomHorizontalFlip
from .oo import VideoResize
from .oo import VideoToTensor

imagenet_stats = dict(mean=(0.43216, 0.394666, 0.37645),
                      std=(0.22803, 0.22145, 0.216989))

__all__ = [
    'Identity', 'OneOf', 'VideoCenterCrop', 'VideoNormalize', 'VideoPath',
    'VideoRandomCrop', 'VideoRandomErase', 'VideoRandomHorizontalFlip',
    'VideoResize', 'VideoToTensor', 'train_tfms', 'valid_tfms'
]
