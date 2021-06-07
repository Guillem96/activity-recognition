from typing import Tuple

import torchvision.transforms as T

import ar
import ar.transforms as VT


def train_tfms(size: Tuple[int, int] = (112, 112),
               do_crop: bool = True,
               do_horizontal_flip: bool = True,
               do_erase: bool = False,
               normalize: bool = True) -> ar.typing.Transform:

    tfms = [VT.VideoToTensor()]

    if do_crop:
        h, w = size
        eh = int(h * 1.3)
        ew = int(w * 1.3)
        tfms.append(VT.VideoResize((eh, ew)))
        tfms.append(VT.VideoRandomCrop(size))
    else:
        tfms.append(VT.VideoResize(size))

    if do_erase:
        tfms.append(VT.VideoRandomErase())

    if do_horizontal_flip:
        tfms.append(VT.VideoRandomHorizontalFlip())

    if normalize:
        tfms.append(VT.VideoNormalize(**VT.imagenet_stats))

    return T.Compose(tfms)


def valid_tfms(size: Tuple[int, int] = (112, 112)) -> ar.typing.Transform:
    h, w = size
    tfms = train_tfms(size=(int(h * 1.3), int(w * 1.3)),
                      do_crop=False,
                      do_horizontal_flip=False,
                      do_erase=False,
                      normalize=True)
    tfms.transforms.append(VT.VideoCenterCrop(size))
    return tfms
