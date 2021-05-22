from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as zoo

from ar.typing import Optimizer

_FEATURE_EXTRACTORS = {
    'resnet18',
    'resnet50',
    'resnet101',
    'densenet121',
    'densenet169',
    'densenet201',
    'densenet161',
    'mobilenet_v2',
    'inception_v3',
}

_VIDEO_FEATURE_EXTRACTORS = {
    'r2plus1d_18',
}


def get_lr(optimizer: Optimizer, reduce: str = 'first') -> float:
    """
    Get the current optimizer's learning rate

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        Optimizer to retrieve the learning rate
    reduce: str
        It can be 'first', 'sum' or 'mean'. Depending on the reduction mode
        the learning rate of the different param_groups will be aggregated
        differently
    
    Returns
    -------
    float
    """
    assert reduce in {'sum', 'mean', 'first'}
    if reduce == 'first':
        return optimizer.param_groups[0]['lr']
    elif reduce == 'sum':
        return sum(o['lr'] for o in optimizer.param_groups)
    else:
        n = len(optimizer.param_groups)
        return sum(o['lr'] for o in optimizer.param_groups) / float(n)


def video_feature_extractor(fe: str,
                            pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Given a architecture name builds the nn.Module outputing the features.

    Parameters
    ----------
    fe : str
        Name of the video CNN architecture
    pretrained : bool, default True
        If the the feature extractor has to be pretrained or not

    Returns
    -------
    Tuple[nn.Module, int]
    """
    assert fe in _VIDEO_FEATURE_EXTRACTORS

    if fe == 'r2plus1d_18':
        r2plus1d_18 = zoo.video.r2plus1d_18(pretrained=pretrained)
        module = nn.Sequential(r2plus1d_18.stem, r2plus1d_18.layer1,
                               r2plus1d_18.layer2, r2plus1d_18.layer3,
                               r2plus1d_18.layer4, r2plus1d_18.avgpool)
        in_features = r2plus1d_18.fc.in_features

    return module, in_features


def image_feature_extractor(fe: str,
                            pretrained: bool = True) -> Tuple[nn.Module, int]:
    """
    Given a architecture name build the nn.Module that outputs the CNN features

    Parameters
    ----------
    fe: str
        Name of the CNN architecture
    pretrained: bool, default True
        If the the feature extractor has to be pretrained or not
    
    Returns
    -------
    Tuple[nn.Module, int]
    """
    assert fe in _FEATURE_EXTRACTORS

    if fe.startswith('resnet'):
        resnet = zoo.__dict__[fe](pretrained=pretrained)
        nn_fe, in_f = (nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu,
                                     resnet.maxpool, resnet.layer1,
                                     resnet.layer2, resnet.layer3,
                                     resnet.layer4,
                                     resnet.avgpool), resnet.fc.in_features)

    elif fe.startswith('densenet'):
        densenet = zoo.__dict__[fe](pretrained=pretrained)
        nn_fe, in_f = (nn.Sequential(densenet.features, nn.ReLU(inplace=True),
                                     nn.AdaptiveAvgPool2d((1, 1))),
                       densenet.classifier.in_features)

    elif fe == 'mobilenet_v2':
        mobilenet = zoo.mobilenet_v2(pretrained=pretrained)
        nn_fe, in_f = (nn.Sequential(mobilenet.features,
                                     nn.AdaptiveAvgPool2d(
                                         (1, 1))), mobilenet.last_channel)
    elif fe == 'inception_v3':
        inception = zoo.inception_v3(pretrained=pretrained)
        nn_fe = nn.Sequential(InceptionInpTransform(), inception.Conv2d_1a_3x3,
                              inception.Conv2d_2a_3x3, inception.Conv2d_2b_3x3,
                              nn.MaxPool2d(kernel_size=3, stride=2),
                              inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
                              nn.MaxPool2d(kernel_size=3,
                                           stride=2), inception.Mixed_5b,
                              inception.Mixed_5c, inception.Mixed_5d,
                              inception.Mixed_6a, inception.Mixed_6b,
                              inception.Mixed_6c, inception.Mixed_6d,
                              inception.Mixed_6e, inception.Mixed_7a,
                              inception.Mixed_7b, inception.Mixed_7c,
                              nn.AdaptiveAvgPool2d((1, 1)))
        in_f = inception.fc.in_features

    return nn_fe, in_f


class InceptionInpTransform(nn.Module):

    def __init__(self) -> None:
        super(InceptionInpTransform, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ch0 = torch.unsqueeze(x[:, 0],
                                1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
        x_ch1 = torch.unsqueeze(x[:, 1],
                                1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
        x_ch2 = torch.unsqueeze(x[:, 2],
                                1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x
