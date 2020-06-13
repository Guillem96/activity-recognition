from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import ar
from ar import utils


################################################################################

class VideoEncoder2D(nn.Module):

    def __init__(self, 
                 feature_extractor: str, 
                 rnn_units: int = 512,
                 bidirectional: bool = True,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False) -> None:
        
        super(VideoEncoder2D, self).__init__()
        
        self.features, in_classifier = utils.nn.image_feature_extractor(
            feature_extractor, pretrained=pretrained)

        for p in self.features.parameters():
            p.requires_grad = not freeze_feature_extractor
        
        self.rnn = nn.LSTM(in_classifier, rnn_units, num_layers=2,
                           bidirectional=bidirectional, dropout=.5,
                           batch_first=True)
        
        hidden_size = rnn_units * (2 if bidirectional else 1)
        
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH)
        b, c, f, h, w = x.size()

        # x: (BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH)
        x = x.permute(0, 2, 1, 3, 4)

        # Rid of from temporal axis 
        # x: (BATCH * FRAMES, CHANNELS, HEIGHT, WIDTH)
        x = x.contiguous().view(-1, c, h, w)

        # Now we are able to feed the input video clips to a CNN
        # x: (BATCH * FRAMES, FEATURES, 1, 1)
        x = self.features(x)

        # Recover temporal axis
        # x: (BATCH, FRAMES, FEATURES)
        x = x.view(b, f, -1)

        # lstm_out: (BATCH, FRAMES, HIDDEN_SIZE)
        lstm_out, _ = self.rnn(x)

        # lstm_out: (FRAMES, BATCH, HIDDEN_SIZE)
        lstm_out = lstm_out.permute(1, 0, 2)
        
        # attn_output: (BATCH, HIDDEN_SIZE)
        attn_output, _  = self.attention(lstm_out, lstm_out, lstm_out)
        return attn_output.sum(0)


class LRCN(utils.checkpoint.SerializableModule):
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
                 freeze_feature_extractor: bool = False) -> None:
        super(LRCN, self).__init__()

        # Frames feature extractor params
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Temporal aware units params
        self.bidirectional = bidirectional
        self.rnn_units = rnn_units

        # Declare nn modules
        self.features = VideoEncoder2D(
            feature_extractor=self.feature_extractor,
            rnn_units=self.rnn_units,
            bidirectional=self.bidirectional,
            pretrained=pretrained,
            freeze_feature_extractor=freeze_feature_extractor)

        hidden_size = self.rnn_units * (2 if self.bidirectional else 1)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(.5)
        self.linear2 = nn.Linear(hidden_size, self.n_classes)

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
        x = self.features(x)
        x = F.relu(self.dropout(self.linear1(x)))
        return self.linear2(x).log_softmax(-1)


class LRCNWithAudio(utils.checkpoint.SerializableModule):
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
                 freeze_feature_extractor: bool = False,
                 
                 # Audio parameters
                 audio_sample_rate: int = 16000,
                 audio_features: int = 512,
                 n_mel_features: int = 40) -> None:
        super(LRCNWithAudio, self).__init__()

        # Frames feature extractor params
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Temporal aware units params
        self.bidirectional = bidirectional
        self.rnn_units = rnn_units

        # Audio features
        self.audio_features = audio_features
        self.audio_sample_rate = audio_sample_rate
        self.n_mel_features = n_mel_features

        # Declare feature extractor for video
        self.features = VideoEncoder2D(
            feature_extractor=self.feature_extractor,
            rnn_units=self.rnn_units,
            bidirectional=self.bidirectional,
            pretrained=pretrained,
            freeze_feature_extractor=freeze_feature_extractor)

        self.audio_fe = ar.audio.MelFrequencyFeatureExtractor(
            feature_extractor=feature_extractor,
            out_features=audio_features,
            mel_features=n_mel_features,
            audio_sample_rate=audio_sample_rate)

        video_output_size = self.rnn_units * (2 if self.bidirectional else 1)
        self.linear1 = nn.Linear(video_output_size + audio_features, 
                                 video_output_size + audio_features)
        self.dropout = nn.Dropout(.5)
        self.linear2 = nn.Linear(video_output_size + audio_features, 
                                 self.n_classes)

    def config(self) -> dict:
        return {
            'feature_extractor': self.feature_extractor,
            'n_classes': self.n_classes,
            'pretrained': False,
            'bidirectional': self.bidirectional,
            'rnn_units': self.rnn_units,
            'freeze_feature_extractor': False,
            'audio_features': self.audio_features,
            'audio_sample_rate': self.audio_sample_rate,
            'n_mel_features': self.n_mel_features
        }

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        video_features = self.features(x[0])
        audio_features = self.audio_fe(x[1])
        x = torch.cat([video_features, audio_features], dim=1)
        x = F.relu(self.dropout(self.linear1(x)))
        return self.linear2(x).log_softmax(-1)
