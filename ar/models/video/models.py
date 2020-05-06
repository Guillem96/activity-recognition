import torch
import torch.nn as nn

from ar import utils


class LRCNN(utils.checkpoint.SerializableModule):
    """
    Model described at Long-term Recurrent Convolutional Networks for Visual 
    Recognition and Description (https://arxiv.org/abs/1411.4389)
    """
    def __init__(self, 
                 feature_extractor: str, 
                 n_classes: int, 
                 rnn_units: int = 512,
                 bidirectional: bool = True,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False):
        super(LRCNN, self).__init__()

        # Frames feature extractor params
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Temporal aware units params
        self.bidirectional = bidirectional
        self.rnn_units = rnn_units

        # Declare nn modules
        self.features, in_classifier = utils.nn.image_feature_extractor(
            feature_extractor, pretrained=True)

        for p in self.features.parameters():
            p.requires_grad = not freeze_feature_extractor
            
        self.rnn = nn.LSTM(in_classifier, self.rnn_units, num_layers=2,
                           bidirectional=self.bidirectional, dropout=.1,
                           batch_first=True)
        
        hidden_size = self.rnn_units * (2 if self.bidirectional else 1)
        
        self.agg_weights = nn.Parameter(
            torch.ones(hidden_size) * (1. / hidden_size))
        
        self.clf = nn.Linear(hidden_size, self.n_classes)

    def config(self) -> dict:
        return {
            'feature_extractor': self.feature_extractor,
            'n_classes': self.n_classes,
            'pretrained': False,
            'bidirectional': self.bidirectional,
            'rnn_units': self.rnn_units,
            'freeze_feature_extractor': False
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH)
        b, c, f, h, w = x.size()

        # x: (BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH)
        x = x.permute(0, 2, 1, 3, 4)

        # Rid of from temporal axis 
        # x: (BATCH * FRAMES, CHANNELS, HEIGHT, WIDTH)
        x = x.contiguous().view(-1, c, h, w)

        # Now we are able to feed the input video clips to a CNN
        # x: (BATCH * FRAMES, FEATURES)
        x = self.features(x)

        # Recover temporal axis
        # x: (BATCH, FRAMES, FEATURES)
        x = x.view(b, f, -1)

        # lstm_out: (BATCH, FRAMES, HIDDEN_SIZE)
        lstm_out, _ = self.rnn(x)
        
        # weighted_sum: (BATCH, HIDDEN_SIZE)
        weighted_sum = (self.agg_weights * lstm_out).sum(1)

        # clf_out: (BATCH, N_CLASSES)
        clf_out = self.clf(weighted_sum)

        return clf_out.log_softmax(-1)
