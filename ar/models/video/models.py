from typing import Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import ar
from ar import utils


################################################################################

class MLP(nn.Module):

    def __init__(self, 
                 features: Sequence[int],
                 batch_norm: bool = True,
                 dropout: float = .3) -> None:
        
        super(MLP, self).__init__()

        def build_block(in_f: int, out_f: int, idx: int) -> nn.Module:
            block = nn.Sequential()
            block.add_module(f'block_{idx}_linear', nn.Linear(in_f, out_f))
            block.add_module(f'block_{idx}_ReLU', nn.ReLU(inplace=True))
            if batch_norm:
                block.add_module(f'block_{idx}_BatchNorm', 
                                 nn.BatchNorm1d(out_f))
            block.add_module(f'block_{idx}_Dropout', nn.Dropout(dropout))
            return block
        
        blocks = []
        for i, (in_f, out_f) in enumerate(zip(features[:-1], features[1:])):
            blocks.append(build_block(in_f, out_f, idx=i))
        
        self.mlp = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class _LRCNEncoder(nn.Module):

    def __init__(self, 
                 feature_extractor: str, 
                 out_features: int,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False) -> None:
        
        super(_LRCNEncoder, self).__init__()
        
        self.features, in_classifier = utils.nn.image_feature_extractor(
            feature_extractor, pretrained=pretrained)
        
        self.pooling = nn.Linear(in_classifier, out_features)
        self.relu = nn.ReLU(inplace=True)

        for p in self.features.parameters():
            p.requires_grad = not freeze_feature_extractor
        
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

        # (BATCH, FRAMES, OUT_FEATURES)
        return self.relu(self.pooling(x))


class _LRCNDecoder(nn.Module):

    def __init__(self, 
                 input_features: int,
                 rnn_units: int,
                 bidirectional: bool,
                 fusion_mode: str = 'sum') -> None:
        
        super(_LRCNDecoder, self).__init__()

        if fusion_mode not in {'sum', 'attn'}:
            raise ValueError(f'fusion_mode must be either sum or attn')

        self.fusion_mode = fusion_mode

        self.rnn = nn.LSTM(input_features, rnn_units, num_layers=2,
                           bidirectional=bidirectional, dropout=.5,
                           batch_first=True)
        
        hidden_size = rnn_units * (2 if bidirectional else 1)
        
        if self.fusion_mode == 'attn':
            self.fusion = nn.MultiheadAttention(hidden_size, num_heads=8)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH, FRAMES, FEATURES)

        # lstm_out: (BATCH, FRAMES, HIDDEN_SIZE)
        lstm_out, _ = self.rnn(x)

        # lstm_out: (FRAMES, BATCH, HIDDEN_SIZE)
        lstm_out = lstm_out.permute(1, 0, 2)
        
        # (BATCH, HIDDEN_SIZE)
        if self.fusion_mode == 'attn':
            attn_out, _ = self.fusion(lstm_out, lstm_out, lstm_out)
            return attn_out.sum(0)
        elif self.fusion_mode == 'sum':
            return lstm_out.sum(0)
        else:
            return lstm_out


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

        self.encoder = _LRCNEncoder(
            feature_extractor=self.feature_extractor,
            out_features=2048,
            pretrained=pretrained,
            freeze_feature_extractor=freeze_feature_extractor)

        self.decoder = _LRCNDecoder(
            input_features=2048,
            rnn_units=self.rnn_units,
            bidirectional=self.bidirectional,
            fusion_mode='sum')

        hidden_size = self.rnn_units * (2 if self.bidirectional else 1)

        self.linear = MLP(features=[hidden_size, 512, self.n_classes], 
                          batch_norm=False,
                          dropout=.5)

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
        x = self.encoder(x)
        x = self.decoder(x)
        return self.linear(x).log_softmax(-1)


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
        self.encoder = _LRCNEncoder(
            feature_extractor=self.feature_extractor,
            out_features=2048,
            pretrained=pretrained,
            freeze_feature_extractor=freeze_feature_extractor)

        self.decoder = _LRCNDecoder(
            input_features=2048,
            rnn_units=self.rnn_units,
            bidirectional=self.bidirectional,
            fusion_mode='sum')

        self.audio_fe = ar.audio.MelFrequencyFeatureExtractor(
            feature_extractor=feature_extractor,
            out_features=audio_features,
            mel_features=n_mel_features,
            audio_sample_rate=audio_sample_rate)

        video_output_size = self.rnn_units * (2 if self.bidirectional else 1)
        self.linear = MLP(features=[video_output_size + audio_features,
                                    512, self.n_classes],
                          batch_norm=False,
                          dropout=.5)

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
        # Extract video features
        video_features = self.encoder(x[0])
        video_features = self.decoder(video_features)

        # Extract audio features
        audio_features = self.audio_fe(x[1])
        
        # Concatenate and merge features
        x = torch.cat([video_features, audio_features], dim=1)
        return self.linear(x).log_softmax(-1)


class SCL(nn.Module):

    def __init__(self, 
                 feature_extractor: str, 
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False) -> None:

        super(SCL, self).__init__()

        featuere_extractor, in_clf = utils.nn.image_feature_extractor(
            feature_extractor, pretrained=pretrained)
        self.out_channels = in_clf

        # Remove last average pooling so we get a 4d tensor (batch, c, h, w)
        # as output
        removed = list(featuere_extractor.children())[:-1]
        self.scl = nn.Sequential(*removed)

        for p in self.scl.parameters():
            p.requires_grad = not freeze_feature_extractor

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # clips: (BATCH, CHANNELS, FRAMES, H, W)
        b, c, f, h, w = clips.size()

        # x: (BATCH, FRAMES, CHANNELS, HEIGHT, WIDTH)
        clips = clips.permute(0, 2, 1, 3, 4)

        # Rid of from temporal axis 
        # clips: (BATCH * FRAMES, CHANNELS, HEIGHT, WIDTH)
        clips = clips.contiguous().view(-1, c, h, w)

        # x: (BATCH * FRAMES, FEATURES, H', W')
        x = self.scl(clips)

        # x: (BATCH, FRAMES, FEATURES, H', W')
        return x.view(b, f, *x.shape[1:])


class TemporalConv(nn.Module):

    def __init__(self, clips_length: int, out_features: int) -> None:
        super(TemporalConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(clips_length, out_features // 2, 3, padding=1),
            nn.BatchNorm2d(out_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.5))

        self.conv2 = nn.Sequential(
            nn.Conv2d(clips_length, out_features // 2, 5, padding=2),
            nn.BatchNorm2d(out_features // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)


class FstCN(utils.checkpoint.SerializableModule):

    def __init__(self, 
                 feature_extractor: str,
                 n_classes: int,
                 scl_features: int = 64,
                 tcl_features: int = 64,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False) -> None:
        super(FstCN, self).__init__()

        # Frames feature extractor params
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Spatial aware hyperparams
        self.scl_features = scl_features

        # Temporal aware hyperparams
        self.tcl_features = tcl_features

        self.scl = SCL(
            feature_extractor=self.feature_extractor,
            pretrained=pretrained,
            freeze_feature_extractor=freeze_feature_extractor)

        scl_out_features = self.scl.out_channels

        # TCL Branch
        self.tcl_conv = nn.Sequential(
            nn.Conv2d(scl_out_features, self.tcl_features, 1, 1),
            nn.BatchNorm2d(self.tcl_features),
            nn.ReLU(inplace=True))
        self.P = nn.Parameter(torch.randn(self.tcl_features, self.tcl_features))
        self.tcl_temp_conv = TemporalConv(self.tcl_features, self.tcl_features)
        self.tcl_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tcl_linear = MLP([self.tcl_features, 
                               self.tcl_features, 
                               self.tcl_features])

        # Get more abstract SCL features branch
        self.xtra_conv = nn.Sequential(
            nn.Conv2d(scl_out_features, self.scl_features, 1, 1),
            nn.BatchNorm2d(self.scl_features),
            nn.ReLU(inplace=True))
        self.xtra_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.xtra_linear = MLP([self.scl_features, 
                                self.scl_features, 
                                self.scl_features])
        self.classifier = nn.Linear(self.scl_features +  self.tcl_features, 
                                    self.n_classes)

    def config(self) -> dict:
        return {
            'feature_extractor': self.feature_extractor,
            'n_classes': self.n_classes,
            'pretrained': False,
            'freeze_feature_extractor': False,
            'scl_features': self.tcl_features,
            'tcl_features': self.tcl_features
        }
    
    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # clips: (BATCH, CHANNELS, FRAMES, H, W)
        
        # x: (BATCH, FRAMES, FEATURES, H', W')
        x = self.scl(clips)

        # Extract abstract features
        b, f, c, h, w = x.size()

        # x: (BATCH * FRAMES, FEATURES, H', W')
        x = x.contiguous().view(-1, c, h, w)

        # xtra_features: (BATCH * FRAMES, FEATURES, H', W')
        xtra_features = self.xtra_conv(x)

        # xtra_features: (BATCH * FRAMES, FEATURES, 1, 1)
        xtra_features = self.xtra_avg_pool(xtra_features)

        # xtra_features: (BATCH * FRAMES, FEATURES)
        xtra_features = xtra_features.view(b * f, -1)
        xtra_features = self.xtra_linear(xtra_features)
        
        # xtra_features: (BATCH, FRAMES, FEATURES)
        xtra_features = xtra_features.view(b, f, -1)
        
        # xtra_features: (BATCH, FEATURES)
        xtra_features = xtra_features.mean(1)

        # tcl_features: (BATCH * FRAMES, FEATURES, H', W')
        tcl_features = self.tcl_conv(x)

        # tcl_features: (BATCH, FRAMES, FEATURES, H' x W')
        tcl_features = tcl_features.view(b, f, self.tcl_features, -1)

        # tcl_features: (BATCH, FRAMES, H' x W', FEATURES')
        tcl_features = tcl_features.permute(0, 1, 3, 2) @ self.P

        # tcl_features: (BATCH, FEATURES', FRAMES, H' x W')
        tcl_features = tcl_features.permute(0, 3, 1, 2)

        # tcl_features: (BATCH, TEMP_FEATURES, FEATURES, H'x W')
        tcl_features = self.tcl_temp_conv(tcl_features)

        # tcl_features: (BATCH, TEMP_FEATURES)
        tcl_features = self.tcl_avg_pool(tcl_features).view(b, -1)
        tcl_features = self.tcl_linear(tcl_features)

        features = torch.cat([xtra_features, tcl_features], dim=1)
        return self.classifier(features).log_softmax(-1)
