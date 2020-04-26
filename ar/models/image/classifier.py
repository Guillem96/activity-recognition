import torch
import torch.nn as nn
import torch.nn.functional as F

from ar.utils.nn import image_feature_extractor


class ImageClassifier(nn.Module):

    def __init__(self, 
                 feature_extractor: str, 
                 n_classes: int, 
                 pretrained: bool = True):
        super(ImageClassifier, self).__init__()

        self.features, in_classifier = image_feature_extractor(
            feature_extractor, pretrained=True)
        self.dropout = nn.Dropout(.5)
        self.classifier = nn.Linear(in_classifier, n_classes)
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x.log_softmax(dim=-1)
