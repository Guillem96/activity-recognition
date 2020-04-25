from .oo import (RandomCrop, CenterCrop, Resize, ToFloatTensorInZeroOne,
                 Normalize, RandomHorizontalFlip, Pad, OneOf, Identity)

imagenet_stats = dict(mean=[0.43216, 0.394666, 0.37645],
                      std=[0.22803, 0.22145, 0.216989])