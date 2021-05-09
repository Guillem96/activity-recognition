import torch
import torch.nn as nn

import ar


class MelFrequencyFeatureExtractor(nn.Module):
    """
    Given a batch of audio samples in time domain generates the corresponding
    MFCC and feeds them to a pretrained CNN to generate features.

    Since, this object is an nn.Module thin can be trained along a downstream
    task, for example audio classification.

    Parameters
    ----------
    feature_extractor: str
        Pretrained feature extractor. see `ar.nn._FEATURE_EXTRACTORS` to see
        the available ones
    out_features: int
        Number of features to generate for an audio sample
    mel_features: int, default 16
        Number of mels to use when extracting MFCCs
    audio_sample_rate: int, default 16000
        Audio sample rate in samples per second
    """

    def __init__(self,
                 feature_extractor: str,
                 out_features: int,
                 mel_features: int = 16,
                 audio_sample_rate: int = 16000) -> None:
        import torchaudio.transforms as AT

        super(MelFrequencyFeatureExtractor, self).__init__()

        self.mfcc = AT.MFCC(audio_sample_rate,
                            mel_features,
                            log_mels=True,
                            melkwargs=dict(n_fft=512,
                                           win_length=400,
                                           hop_length=160))

        self.channels_up = nn.Conv2d(1, 3, kernel_size=1, stride=1)

        # TODO: Should we provide the possibility of freezing this?
        self.features, in_next = ar.utils.nn.image_feature_extractor(
            feature_extractor, pretrained=True)

        self.features_adjust = nn.Linear(in_next, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mfcc: (BATCH, 1, HEIGHT, WIDTH)
        mfcc = self.mfcc(x)

        # up_channels: (BATCH, 3, HEIGHT, WIDTH)
        up_channels = self.channels_up(mfcc)

        # features: (BATCH, CNN_FEATURES, 1, 1)
        features = self.features(up_channels)

        # features: (BATCH, CNN_FEATURES)
        features = torch.flatten(features, 1)

        # return value: (BATCH, OUT_FEATURES)
        return self.features_adjust(features)
