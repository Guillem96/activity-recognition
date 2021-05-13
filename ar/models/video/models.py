from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import ar

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


################################################################################


class _LRCNEncoder(nn.Module):

    def __init__(self,
                 feature_extractor: str,
                 out_features: int,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False) -> None:

        super(_LRCNEncoder, self).__init__()

        self.features, in_classifier = ar.nn.image_feature_extractor(
            feature_extractor, pretrained=pretrained)

        self.pooling = nn.Linear(in_classifier, out_features)
        self.relu = nn.ReLU(inplace=True)

        for p in self.features.parameters():
            p.requires_grad = not freeze_feature_extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH, CHANNELS, FRAMES, HEIGHT, WIDTH)

        # x: (BATCH, FRAMES, FEATURES)
        x = _frame_level_forward(x, self.features).squeeze()

        # (BATCH, FRAMES, OUT_FEATURES)
        return self.relu(self.pooling(x))


class _LRCNDecoder(nn.Module):

    def __init__(self,
                 input_features: int,
                 rnn_units: int,
                 bidirectional: bool,
                 fusion_mode: str = 'sum') -> None:

        super(_LRCNDecoder, self).__init__()

        if fusion_mode not in {'sum', 'attn', 'avg', 'last'}:
            raise ValueError(f'fusion_mode must be either sum or attn')

        self.fusion_mode = fusion_mode

        self.rnn = nn.LSTM(input_features,
                           rnn_units,
                           num_layers=2,
                           bidirectional=bidirectional,
                           dropout=.5,
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
        elif self.fusion_mode == 'avg':
            return lstm_out.mean(0)
        elif self.fusion_mode == 'last':
            return lstm_out[-1]
        else:
            return lstm_out


class LRCN(ar.utils.checkpoint.SerializableModule):
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
                 fusion_mode: str = 'sum') -> None:
        super(LRCN, self).__init__()

        # Frames feature extractor params
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Temporal aware units params
        self.bidirectional = bidirectional
        self.rnn_units = rnn_units
        self.fusion_mode = fusion_mode

        self.encoder = _LRCNEncoder(
            feature_extractor=self.feature_extractor,
            out_features=2048,
            pretrained=pretrained,
            freeze_feature_extractor=freeze_feature_extractor)

        self.decoder = _LRCNDecoder(input_features=2048,
                                    rnn_units=self.rnn_units,
                                    bidirectional=self.bidirectional,
                                    fusion_mode=fusion_mode)

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
            'freeze_feature_extractor': False,
            'fusion_mode': self.fusion_mode
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return self.linear(x).log_softmax(-1)


################################################################################


class TemporalConv(nn.Module):

    def __init__(self, clips_length: int, out_features: int) -> None:
        super(TemporalConv, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(clips_length, out_features // 2, 3, padding=1),
            nn.BatchNorm2d(out_features // 2), nn.ReLU(inplace=True),
            nn.Dropout2d(.5))

        self.conv2 = nn.Sequential(
            nn.Conv2d(clips_length, out_features // 2, 5, padding=2),
            nn.BatchNorm2d(out_features // 2), nn.ReLU(inplace=True),
            nn.Dropout2d(.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)


class FstCN(ar.utils.checkpoint.SerializableModule):
    """Implements the model defined at Human Action Recognition using 
    Factorized Spatio-Temporal Convolutional Networks 
    (https://arxiv.org/pdf/1510.00562.pdf)

    Parameters
    ----------
    feature_extractor: str
        Feature extractor that will be used as a backbone
    n_classes: int
        Output classes
    st: int
        Stride to sample frames from the input clip
    dt: int
        Offset to compute the V^{diff}
    scl_features: int, defaults 64
        Features for the spatial branch
    tcl_features: int, defaults 64
        Features for the temporal branch
    pretrained: bool, defaults True
        Initialize the backbone from a torch checkpoint?
    freeze_feature_extractor: bool, False
        Whether or not to freeze the backbone.
    """

    def __init__(self,
                 feature_extractor: str,
                 n_classes: int,
                 st: int = 5,
                 dt: int = 9,
                 scl_features: int = 64,
                 tcl_features: int = 64,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False) -> None:
        super(FstCN, self).__init__()

        # Frames feature extractor params
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained

        # Vdiff hyperparams
        self.st = st
        self.dt = dt

        # Spatial aware hyperparams
        self.scl_features = scl_features

        # Temporal aware hyperparams
        self.tcl_features = tcl_features

        # Create SCL. To do so, we instantiate a pretrained image classifier
        # and remove the last average pooling
        feature_extractor, scl_out_features = ar.nn.image_feature_extractor(
            feature_extractor, pretrained=pretrained)

        removed = list(feature_extractor.children())[:-1]
        self.scl = nn.Sequential(*removed)

        for p in self.scl.parameters():
            p.requires_grad = not freeze_feature_extractor

        # TCL Branch
        self.tcl_conv = nn.Sequential(
            nn.Conv2d(scl_out_features, self.tcl_features, 1, 1),
            nn.BatchNorm2d(self.tcl_features), nn.ReLU(inplace=True))
        self.P = nn.Parameter(torch.randn(self.tcl_features, self.tcl_features))

        # H x W of reduced Vdiff clips are 4 and 4 respectively for clips of
        # 112x112
        self.tcl_temp_conv = TemporalConv(4 * 4, self.tcl_features)
        self.tcl_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tcl_linear = MLP([self.tcl_features, 2048, self.tcl_features])

        # Get more abstract SCL features branch
        self.xtra_conv = nn.Sequential(
            nn.Conv2d(scl_out_features, self.scl_features, 1, 1),
            nn.BatchNorm2d(self.scl_features), nn.ReLU(inplace=True))
        self.xtra_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.xtra_linear = MLP([self.scl_features, 2048, self.scl_features])

        self.classifier = nn.Linear(self.scl_features + self.tcl_features,
                                    self.n_classes)

    def config(self) -> dict:
        return {
            'feature_extractor': self.feature_extractor,
            'n_classes': self.n_classes,
            'pretrained': False,
            'freeze_feature_extractor': False,
            'scl_features': self.tcl_features,
            'tcl_features': self.tcl_features,
            'st': self.st,
            'dt': self.dt,
        }

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # clips: (BATCH, CHANNELS, FRAMES, H, W)
        # Temporal stride clips
        V = clips[:, :, ::self.st, ...]
        b, _, f, *_ = V.size()
        clips_length = f - self.dt

        v_idx = torch.arange(clips_length).to(V.device)
        v_idx_offset = v_idx + self.dt

        # Vdiff_features: (BATCH, CHANNELS, FRAMES, H, W)
        Vdiff = V[:, :, v_idx] - V[:, :, v_idx_offset]
        V = V[:, :, v_idx]

        # For spatial clips, when training we sample a single random,
        # when testing we sample the middle frame
        if self.training:
            sampled_frames_idx = torch.randint(high=clips_length, size=(b,))
            sampled_clips = V[torch.arange(b), :, sampled_frames_idx]
        else:
            middle_idx = clips_length // 2
            sampled_clips = V[torch.arange(b), :, middle_idx]

        # single_clip_features: (BATCH, FEATURES, H', W')
        single_clip_features = self.scl(sampled_clips)

        # Vdiff_features: (BATCH, FRAMES, FEATURES, H', W')
        Vdiff_features = _frame_level_forward(Vdiff, self.scl)

        # XTRA Brach
        # xtra_features: (BATCH, FEATURES, H', W')
        xtra_features = self.xtra_conv(single_clip_features)

        # xtra_features: (BATCH, FEATURES, 1, 1)
        xtra_features = self.xtra_avg_pool(xtra_features)

        # xtra_features: (BATCH, FEATURES)
        xtra_features = xtra_features.view(b, -1)
        xtra_features = self.xtra_linear(xtra_features)

        # Temporal Branch
        # tcl_features: (BATCH, FRAMES, FEATURES, H' x W')
        tcl_features = _frame_level_forward(
            Vdiff_features.permute(0, 2, 1, 3, 4), self.tcl_conv)
        tcl_channels = tcl_features.size(2)
        tcl_features = tcl_features.view(b, clips_length, tcl_channels, -1)

        # tcl_features: (BATCH, H' x W', FRAMES, FEATURES')
        tcl_features = tcl_features.permute(0, 3, 1, 2) @ self.P
        print(tcl_features.size())

        # tcl_features: (BATCH, TEMP_FEATURES, H', W')
        tcl_features = self.tcl_temp_conv(tcl_features)

        # tcl_features: (BATCH, TEMP_FEATURES)
        tcl_features = self.tcl_avg_pool(tcl_features).view(b, -1)
        tcl_features = self.tcl_linear(tcl_features)

        features = torch.cat([xtra_features, tcl_features], dim=1)
        return self.classifier(features).log_softmax(-1)


################################################################################


def _video_for_frame_level_fw(video: torch.Tensor) -> torch.Tensor:
    """Reshapes a video tensor so it can be feed at frame level to a spatial 
    CNN"""
    b, c, f, h, w = video.size()
    video = video.permute(0, 2, 1, 3, 4)
    video = video.contiguous().view(b * f, c, h, w)
    return video


def _frame_level_forward(video: torch.Tensor,
                         module: nn.Module) -> torch.Tensor:
    """Given a video and a module, time distributes the module over
       the temporal axis. Feeds frames individually to a spatial CNN
       taking advantage of batch processing.
    """
    # x: (BATCH, CHANNELS, FRAMES, H, W)
    b, _, f, *_ = video.size()

    # x: (BATCH * FRAMES, CHANNELS, H, W)
    x = _video_for_frame_level_fw(video)

    # x: (BATCH, FRAMES, ...)
    x = module(x)

    return x.view(b, f, *x.shape[1:])
