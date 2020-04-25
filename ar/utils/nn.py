import torch
import torch.nn as nn
import torchvision.models as zoo


_FEATURE_EXTRACTORS = {
    'resnet18', 'resnet50', 'resnet101', 
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'mobilenet_v2'
}
    


def image_feature_extractor(fe: str, pretrained: bool = True) -> nn.Module:
    """
    Given a architecture name build the nn.Module that outputs the CNN features

    Parameters
    ----------
    fe: str
        Name of the CNN architecture
    """
    assert fe in _FEATURE_EXTRACTORS

    if fe.startswith('resnet'):
        resnet = zoo.__dict__[fe](pretrained=pretrained)
        return nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool)
    elif fe.startswith('densenet'):
        densenet = zoo.__dict__[fe](pretrained=pretrained)
        return nn.Sequential(
            densenet.features, nn.ReLU(inplace=True), 
            nn.AdaptiveAvgPool2d((1, 1)))
    elif fe == 'mobilenet_v2':
        mobilenet = zoo.mobilenet_v2(pretrained=pretrained)
        return nn.Sequential(
            mobilenet.features, nn.AdaptiveAvgPool2d((1, 1)))
        