from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as zoo

from ar.typing import Optimizer

_FEATURE_EXTRACTORS = {
    'resnet18', 'resnet50', 'resnet101', 
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'mobilenet_v2'
}


def get_lr(optimizer: Optimizer, 
           reduce: str = 'first') -> float:
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
        nn_fe, in_f = (nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool), resnet.fc.in_features)

    elif fe.startswith('densenet'):
        densenet = zoo.__dict__[fe](pretrained=pretrained)
        nn_fe, in_f =  (nn.Sequential(
            densenet.features, nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1))), densenet.classifier.in_features)

    elif fe == 'mobilenet_v2':
        mobilenet = zoo.mobilenet_v2(pretrained=pretrained)
        nn_fe, in_f = (nn.Sequential(mobilenet.features, 
                                      nn.AdaptiveAvgPool2d((1, 1))),
                mobilenet.last_channel)
    
    return nn_fe, in_f
