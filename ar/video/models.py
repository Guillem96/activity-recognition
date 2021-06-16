from typing import Any
from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import torch
import torch.nn as nn
import torchvision.models.video.resnet as resnet3d

import ar

################################################################################


class _MLP(nn.Module):

    def __init__(self,
                 features: Sequence[int],
                 batch_norm: bool = True,
                 dropout: float = .3) -> None:

        super(_MLP, self).__init__()

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
        x = _frame_level_forward(x, self.features).flatten(2)

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
                           dropout=.3,
                           batch_first=True)

        hidden_size = rnn_units * (2 if bidirectional else 1)

        if self.fusion_mode == 'attn':
            self.fusion = nn.MultiheadAttention(hidden_size,
                                                num_heads=8,
                                                dropout=.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (BATCH, FRAMES, FEATURES)

        # lstm_out: (BATCH, FRAMES, HIDDEN_SIZE)
        lstm_out, _ = self.rnn(x)

        # (BATCH, HIDDEN_SIZE)
        if self.fusion_mode == 'attn':
            # lstm_out: (FRAMES, BATCH, HIDDEN_SIZE)
            lstm_out = lstm_out.permute(1, 0, 2)
            attn_out, _ = self.fusion(lstm_out, lstm_out, lstm_out)
            return attn_out.sum(0)
        elif self.fusion_mode == 'sum':
            return lstm_out.sum(1)
        elif self.fusion_mode == 'avg':
            return lstm_out.mean(1)
        elif self.fusion_mode == 'last':
            return lstm_out[:, -1]
        else:
            return lstm_out


class LRCN(ar.utils.checkpoint.SerializableModule):
    """
    Model described at Long-term Recurrent Convolutional Networks for Visual 
    Recognition and Description (https://arxiv.org/abs/1411.4389).

    .. image:: ../images/ar-lrcn.png
        :width: 400

    Parameters
    ----------
    feature_extractor: str
        Model architecture to use as a encoder. 
        See `ar.nn.image_feature_extractor`
    n_classes: int
        Number of NN outputs
    rnn_units: int, defaults 512
        Neurons used in the decoder. Note that if `bidirectional` is True
        this value will be doubled.
    bidirectional: bool, defaults True
        Use a bidirectional LSTM at the decoder
    pretrained: bool, defaults True
        Use a pretrained encoder
    freeze_feature_extractor: bool, defaults False
        Requires grad set to false for the encoder.
    fusion_mode: str, defaults 'sum'
        Method to fuse the outputs of the decoder (sum, avg, attn or last)
    """

    def __init__(self,
                 feature_extractor: str,
                 n_classes: int,
                 rnn_units: int = 512,
                 bidirectional: bool = True,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False,
                 dropout: float = .3,
                 fusion_mode: str = 'sum') -> None:
        super(LRCN, self).__init__()

        # Frames feature extractor params
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.freeze_feature_extractor = freeze_feature_extractor

        # Temporal aware units params
        self.bidirectional = bidirectional
        self.rnn_units = rnn_units
        self.fusion_mode = fusion_mode
        self.dropout = dropout

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

        self.linear = _MLP(features=[hidden_size, 512],
                           batch_norm=False,
                           dropout=dropout)

        self.clf = nn.Linear(512, self.n_classes)

    def config(self) -> Dict[str, Any]:
        return {
            'feature_extractor': self.feature_extractor,
            'n_classes': self.n_classes,
            'pretrained': self.pretrained,
            'bidirectional': self.bidirectional,
            'rnn_units': self.rnn_units,
            'freeze_feature_extractor': self.freeze_feature_extractor,
            'fusion_mode': self.fusion_mode,
            'dropout': self.dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.linear(x)
        return self.clf(x).log_softmax(-1)


################################################################################


class _TemporalConv(nn.Module):

    def __init__(self, clips_length: int, out_features: int) -> None:
        super(_TemporalConv, self).__init__()

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

    NOTE: Only works with video clips of 112x112

    .. image:: ../images/ar-fstcn.png
        :width: 400

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
                 dropout: float = .5,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False) -> None:
        super(FstCN, self).__init__()

        # Frames feature extractor params
        self.feature_extractor = feature_extractor
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.freeze_feature_extractor = freeze_feature_extractor

        # Vdiff hyperparams
        self.st = st
        self.dt = dt

        # Spatial aware hyperparams
        self.scl_features = scl_features

        # Temporal aware hyperparams
        self.tcl_features = tcl_features

        self.dropout = dropout

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
        self.tcl_temp_conv = _TemporalConv(4 * 4, self.tcl_features)
        self.tcl_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.tcl_linear = _MLP([self.tcl_features, 2048, self.tcl_features],
                               dropout=self.dropout)

        # Get more abstract SCL features branch
        self.xtra_conv = nn.Sequential(
            nn.Conv2d(scl_out_features, self.scl_features, 1, 1),
            nn.BatchNorm2d(self.scl_features), nn.ReLU(inplace=True))
        self.xtra_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.xtra_linear = _MLP([self.scl_features, 2048, self.scl_features],
                                dropout=self.dropout)

        self.classifier = nn.Linear(self.scl_features + self.tcl_features,
                                    self.n_classes)

    def config(self) -> Dict[str, Any]:
        return {
            'feature_extractor': self.feature_extractor,
            'n_classes': self.n_classes,
            'pretrained': self.pretrained,
            'freeze_feature_extractor': self.freeze_feature_extractor,
            'scl_features': self.tcl_features,
            'tcl_features': self.tcl_features,
            'st': self.st,
            'dt': self.dt,
            'dropout': self.dropout
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

        # tcl_features: (BATCH, TEMP_FEATURES, H', W')
        tcl_features = self.tcl_temp_conv(tcl_features)

        # tcl_features: (BATCH, TEMP_FEATURES)
        tcl_features = self.tcl_avg_pool(tcl_features).view(b, -1)
        tcl_features = self.tcl_linear(tcl_features)

        features = torch.cat([xtra_features, tcl_features], dim=1)
        return self.classifier(features).log_softmax(-1)


################################################################################


class R2plus1_18(ar.utils.checkpoint.SerializableModule):
    """Implements the model defined at A Closer Look at Spatiotemporal 
    Convolutions for Action Recognition (https://arxiv.org/abs/1711.11248).

    .. image:: ../images/r2-1.png
        :width: 500

    Parameters
    ----------
    n_classes: int
        Number of outputs of the architecture.
    dropout: float, defaults .3
        Dropout rate of the last layer.
    pretrainedL bool, defaults True
        Use the pretrained weights of kinetics-400
    freeze_feature_extractor: bool, defaults False
        All parameters requires grad set to false except from the last linear
        layer. 
    """

    def __init__(self,
                 n_classes: int,
                 dropout: float = .3,
                 pretrained: bool = True,
                 freeze_feature_extractor: bool = False) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.pretrained = pretrained
        self.freeze_feature_extractor = freeze_feature_extractor

        fe, in_features = ar.nn.video_feature_extractor('r2plus1d_18',
                                                        pretrained=True)
        for p in fe.parameters():
            p.requires_grad = not freeze_feature_extractor

        self.fe = fe
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(in_features, n_classes)

    def config(self) -> dict:
        return {
            'freeze_feature_extractor': self.freeze_feature_extractor,
            'n_classes': self.n_classes,
            'pretrained': self.pretrained,
        }

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        x = self.fe(video).flatten(1)
        x = self.dropout(x)
        return self.classifier(x).log_softmax(-1)


################################################################################

_BOTTLENECK_EXPANSION = 4
_BlockCls = Union[Type['_NoDegenTempBottleNeck'], Type['_Bottleneck']]
_3DKernel = Union[int, Tuple[int, int, int]]


class _Bottleneck(nn.Module):

    def __init__(self,
                 inplanes: int,
                 planes: int,
                 conv_builder: nn.Module = resnet3d.Conv3DSimple,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 non_temp_degen: bool = False) -> None:

        super().__init__()

        # 1x1x1
        ks: _3DKernel = (3, 1, 1) if non_temp_degen else 1
        pad: _3DKernel = (1, 0, 0) if non_temp_degen else 0

        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=ks,
                      padding=pad, bias=False), nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True))

        # Second kernel
        self.conv2 = nn.Sequential(conv_builder(planes, planes, stride=stride),
                                   nn.BatchNorm3d(planes),
                                   nn.ReLU(inplace=True))

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes,
                      planes * _BOTTLENECK_EXPANSION,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm3d(planes * _BOTTLENECK_EXPANSION))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


class _NoDegenTempBottleNeck(_Bottleneck):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['non_temp_degen'] = False
        super().__init__(*args, **kwargs)


class _TimeToChannelFusion(nn.Module):

    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, slow_x: torch.Tensor,
                fast_x: torch.Tensor) -> torch.Tensor:
        # slow_x: (BATCH, C, T, H, W)
        # fast_x: (BATCH, βC, αT, H, W)
        b, c, t, h, w = slow_x.size()

        # fast_x: (BATCH, βαC, T, H, W)
        fast_x = fast_x.view(b, int(self.alpha * self.beta * c), t, h, w)

        return slow_x + fast_x


class _TimeStridedSampleFuse(nn.Module):

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = int(alpha)

    def forward(self, slow_x: torch.Tensor,
                fast_x: torch.Tensor) -> torch.Tensor:
        fast_selected_frames = fast_x[:, :, ::self.alpha]
        return torch.cat([slow_x, fast_selected_frames], dim=1)


class _TimeStridedConvFuse(nn.Module):

    def __init__(self, slow_features: int, alpha: int, beta: float) -> None:
        super().__init__()
        fast_features = int(slow_features * beta)

        # Differs from paper description.
        # Time strided conv to downsample 32 frames from fast pathway to
        # 4 frames of slow path
        self.conv = nn.Sequential(
            nn.Conv3d(fast_features,
                      fast_features * 2,
                      kernel_size=(5, 1, 1),
                      stride=(alpha, 1, 1)), nn.BatchNorm3d(fast_features * 2),
            nn.ReLU(inplace=True))

    def forward(self, slow_x: torch.Tensor,
                fast_x: torch.Tensor) -> torch.Tensor:
        # slow_x: (BATCH, C, T, H, W)
        # fast_x: (BATCH, βC, αT, H, W)
        fast_x = self.conv(fast_x)
        return torch.cat([slow_x, fast_x], dim=1)


class _FusionFastSlow(nn.Module):

    def __init__(self, fusion_mode: str, in_features: int, alpha: int,
                 beta: float) -> None:
        super().__init__()

        self.fuser: Union[_TimeToChannelFusion, _TimeStridedSampleFuse,
                          _TimeStridedConvFuse]
        if fusion_mode == 'time-to-channel':
            self.fuser = _TimeToChannelFusion(alpha, beta)
        elif fusion_mode == 'time-strided-sample':
            self.fuser = _TimeStridedSampleFuse(alpha)
        elif fusion_mode == 'time-strided-conv':
            self.fuser = _TimeStridedConvFuse(in_features, alpha, beta)
        else:
            raise ValueError(f'Invalid fusion mode {fusion_mode}. '
                             'Choose one of ["time-to-channel", '
                             '"time-strided-sample", "time-strided-conv"]')

    def forward(self, slow_x: torch.Tensor,
                fast_x: torch.Tensor) -> torch.Tensor:
        return self.fuser(slow_x, fast_x)


class _PathWay(nn.Module):

    def __init__(self,
                 blocks_classes: Sequence[_BlockCls],
                 alpha: int = 8,
                 beta: float = 1 / 8,
                 strides: Sequence[int] = (1, 2, 2, 2),
                 is_slow: bool = False,
                 fusion_mode: Optional[str] = None) -> None:

        super().__init__()

        if len(blocks_classes) != 4:
            raise ValueError(
                "PathWay has 4 layers. "
                f"You provided {len(blocks_classes)} block classes")

        if is_slow and not fusion_mode:
            raise ValueError('You have to provide a fusion_mode if '
                             'is_slow=True')
        elif not is_slow and fusion_mode:
            raise ValueError('is_slow=False (Fastpathway) is not compatible '
                             'with fusion_mode.')

        self.alpha = alpha
        self.beta = beta
        self.fusion_mode = fusion_mode
        self.is_slow = is_slow

        # Features of each layer
        # If we are in a fast pathway we downscale the features by beta
        features_scale = 1 if is_slow else beta
        features = [int(o * features_scale) for o in [64, 128, 256, 256]]
        self._output_features = features[-1] * _BOTTLENECK_EXPANSION

        # Fusion features
        self.inplanes = int(64 * features_scale)

        if fusion_mode:
            # Compute the features for residual connections
            # Note that for the first residual connection we do not have to
            # take into account the bottleneck expansion.
            fusion_features = [
                int(_BOTTLENECK_EXPANSION * f) for f in features[:-1]
            ]
            fusion_features = [self.inplanes] + fusion_features

            self.fusers = nn.ModuleList([
                _FusionFastSlow(fusion_mode, f, alpha, beta)
                for f in fusion_features
            ])
        else:
            # No track fusion features in fast path
            fusion_features = [0] * len(features)

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=3,
                      out_channels=int(64 * features_scale),
                      kernel_size=(1, 7, 7),
                      stride=(1, 2, 2),
                      padding=(0, 3, 3)),
            nn.MaxPool3d(kernel_size=(1, 3, 3),
                         stride=(1, 2, 2),
                         padding=(0, 1, 1)))

        # Residual layers
        # If the fusion mode concatenates instead of sum, the in_features
        # for the subsequent layers change
        layers_extra_inp_feat = [
            self._compute_extra_input(ff) for ff in fusion_features
        ]
        self.layers = nn.ModuleList([
            self._make_layer(
                block_cls=blocks_classes[i],
                features=features[i],
                blocks=2,
                stride=strides[i],
                extra_inp_features=layers_extra_inp_feat[i],
            ) for i in range(4)
        ])

    def _compute_extra_input(self, fusion_features: int) -> int:
        if self.is_slow and self.fusion_mode == 'time-strided-sample':
            return int(self.beta * fusion_features)
        elif self.is_slow and self.fusion_mode == 'time-strided-conv':
            return int(self.beta * 2 * fusion_features)
        else:
            return 0

    @property
    def output_features(self) -> int:
        return self._output_features

    def forward(
        self,
        x: torch.Tensor,
        residuals: Optional[Sequence[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Sequence[torch.Tensor]]]:
        x = self.stem(x)
        laterals = [x]

        if residuals:
            x = self.fusers[0](x, residuals[0])

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if residuals and (i + 1) < len(residuals):
                x = self.fusers[i + 1](x, residuals[i + 1])
            laterals.append(x)

        return x, laterals[:-1]

    def _make_layer(self,
                    block_cls: _BlockCls,
                    features: int,
                    blocks: int,
                    stride: int = 1,
                    extra_inp_features: int = 0) -> nn.Module:
        downsample = None

        if stride != 1 or self.inplanes != features * _BOTTLENECK_EXPANSION:
            ds_stride = resnet3d.Conv3DNoTemporal.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes + extra_inp_features,
                          features * _BOTTLENECK_EXPANSION,
                          kernel_size=1,
                          stride=ds_stride,
                          bias=False),
                nn.BatchNorm3d(features * _BOTTLENECK_EXPANSION))

        layers = [
            block_cls(self.inplanes + extra_inp_features, features,
                      resnet3d.Conv3DNoTemporal, stride, downsample)
        ]

        self.inplanes = features * _BOTTLENECK_EXPANSION
        layers += [
            block_cls(self.inplanes, features, resnet3d.Conv3DNoTemporal)
            for _ in range(1, blocks)
        ]
        return nn.Sequential(*layers)


class SlowFast(ar.utils.checkpoint.SerializableModule):
    """Implementation of the SlowFast model.

    Model described in the article "SlowFast Networks for Video Recognition"
    (https://arxiv.org/pdf/1812.03982.pdf).

    .. image:: ../images/slowfast.png
        :width: 400

    Parameters
    ----------
    n_classes: int
        Number of model outputs to perform classification.
    alpha: float, defaults 8
        Rate of frames of fast path with respect the slow one.
    beta: float, defaults 1/8
        Proportion of features of fast pathway with respect the slow one.
    tau: int, defaults 16
        Stride to sample frames for slow path.
    dropout: float, default .3
        Prob of dropout applied after concatenating the 
        features from both paths.
    fusion_mode: str, defaults time-strided-conv'
        Fusion method for the residual connections from fast to slow path.
        Either 'time-to-channel', 'time-strided-sample' or 'time-strided-conv'.
    """

    def __init__(self,
                 n_classes: int,
                 alpha: int = 8,
                 beta: float = 1 / 8,
                 tau: int = 16,
                 dropout: float = .3,
                 fusion_mode: str = 'time-strided-conv') -> None:

        super().__init__()

        self.n_classes = n_classes

        self.tau = tau
        self.beta = beta
        self.alpha = alpha
        self.fusion_mode = fusion_mode

        self.slow_stride = self.tau
        self.fast_stride = int(self.tau // self.alpha)

        slow_blocks = [
            _Bottleneck, _Bottleneck, _NoDegenTempBottleNeck,
            _NoDegenTempBottleNeck
        ]

        fast_blocks = [
            _NoDegenTempBottleNeck, _NoDegenTempBottleNeck,
            _NoDegenTempBottleNeck, _NoDegenTempBottleNeck
        ]

        self.slowpath = _PathWay(slow_blocks,
                                 alpha=self.alpha,
                                 beta=self.beta,
                                 is_slow=True,
                                 fusion_mode=self.fusion_mode)
        self.fastpath = _PathWay(fast_blocks,
                                 alpha=self.alpha,
                                 beta=self.beta,
                                 is_slow=False)

        self._out_features = (self.slowpath.output_features +
                              self.fastpath.output_features)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self._out_features, n_classes)

    def config(self) -> Dict[str, Any]:
        return {
            'n_classes': self.n_classes,
            'alpha': self.alpha,
            'beta': self.beta,
            'tau': self.tau,
            'fusion_mode': self.fusion_mode,
        }

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        slow_video = video[:, :, ::self.slow_stride]
        fast_video = video[:, :, ::self.fast_stride]

        fast_out, laterals = self.fastpath(fast_video)
        slow_out, _ = self.slowpath(slow_video, laterals)

        pooled_fast = self.avg_pool(fast_out)
        pooled_slow = self.avg_pool(slow_out)

        pooled_features = torch.cat([pooled_fast, pooled_slow], dim=1)
        pooled_features = pooled_features.view(-1, self._out_features)
        pooled_features = self.dropout(pooled_features)

        return self.linear(pooled_features).log_softmax(-1)


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
