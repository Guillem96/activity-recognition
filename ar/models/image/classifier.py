import torch
import torch.nn as nn
import torch.nn.functional as F

from ar.utils.checkpoint import SerializableModule
from ar.utils.nn import image_feature_extractor


class ImageClassifier(SerializableModule):

    def __init__(self,
                 feature_extractor: str,
                 n_classes: int,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False):
        super(ImageClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained

        self.features, in_classifier = image_feature_extractor(
            feature_extractor, pretrained=True)

        for p in self.features.parameters():
            p.requires_grad = not freeze_feature_extractor

        self.dropout = nn.Dropout(.5)
        self.classifier = nn.Linear(in_classifier, n_classes)

    def config(self) -> dict:
        return {
            'feature_extractor': self.feature_extractor,
            'n_classes': self.n_classes,
            'pretrained': self.pretrained
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x.log_softmax(dim=-1)
