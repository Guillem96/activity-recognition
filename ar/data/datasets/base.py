import abc
import warnings
from pathlib import Path
from typing import Collection
from typing import Mapping
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets.video_utils import VideoClips

from ar.typing import Transform

_ClipDatasetSample = Tuple[torch.Tensor, torch.Tensor, int, dict]


class ClipLevelDataset(data.Dataset, abc.ABC):
    def __init__(
        self,
        root: Union[Path, str],
        split: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: int = None,
        transform: Transform = None,
        num_workers: int = 4,
        extensions: Collection[str] = ('mp4', 'avi')
    ) -> None:

        self.root = Path(root)
        self.split = split
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.frame_rate = frame_rate
        self.transform = transform
        self._num_workers = num_workers
        self._extensions = extensions

        self._classes = None
        self._class_2_idx = None

    def _build_video_clips(self) -> VideoClips:
        return VideoClips(list(map(str, self.paths)),
                          self.frames_per_clip,
                          self.step_between_clips,
                          self.frame_rate,
                          None,
                          num_workers=self._num_workers,
                          _video_width=0,
                          _video_height=0,
                          _video_min_dimension=0,
                          _audio_samples=0,
                          _audio_channels=0)

    @abc.abstractproperty
    def split_root(self) -> Path:
        raise NotImplementedError()

    @abc.abstractproperty
    def paths(self) -> Sequence[Path]:
        raise NotImplementedError()

    @abc.abstractproperty
    def labels(self) -> Sequence[str]:
        raise NotImplementedError()

    @property
    def classes(self) -> Sequence[str]:
        if self._classes:
            return self._classes
        self._classes = sorted(list(set(self.labels)))
        return self._classes

    @property
    def class_2_idx(self) -> Mapping[str, int]:
        if self._class_2_idx:
            return self._class_2_idx
        self._class_2_idx = dict(enumerate(self.classes))
        return self._class_2_idx

    @property
    def metadata(self) -> dict:
        return self.video_clips.metadata

    @abc.abstractproperty
    def video_clips(self) -> VideoClips:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> _ClipDatasetSample:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.class_2_idx[self.labels[video_idx]]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, info


class VideoLevelDataset(data.Dataset, abc.ABC):
    def __init__(self,
                 video_paths: Sequence[Union[str, Path]],
                 labels: Sequence[str],
                 frame_rate: int = None,
                 transform: Transform = None) -> None:

        self.video_paths = [str(o) for o in video_paths]
        self.labels = labels
        self.classes = sorted(list(set(labels)))
        self.class_2_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels_ids = [self.class_2_idx[o] for o in self.labels]

        self.frame_rate = frame_rate
        self.transform = transform

    def resample_video(self, video: torch.Tensor,
                       original_fps: int) -> torch.Tensor:

        if self.frame_rate is None:
            return video

        step = float(original_fps) / self.frame_rate
        if step.is_integer():
            step = int(step)
            return video[::step]

        idxs = torch.arange(video.size(0), dtype=torch.float32) * step
        idxs = idxs.floor().to(torch.int64)
        return video[idxs]

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        video, audio, info = torchvision.io.read_video(self.video_paths[idx],
                                                       pts_unit='sec')

        if video.nelement() == 0:
            warnings.warn(f'Error loading video {self.video_paths[idx]}')
            return video, audio, info

        if 'video_fps' not in info:
            # raise ValueError(f'Error fetching metadata from'
            #                  f' {self.video_paths[idx]} video')
            pass
        else:
            video = self.resample_video(video, info['video_fps'])

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, self.class_2_idx[self.labels[idx]]
