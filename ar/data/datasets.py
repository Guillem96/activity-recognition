import abc
from pathlib import Path
from typing import Tuple, Collection, Union, Sequence

import torch
import torch.utils.data as data

from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

from ar.typing import Transform


class VideoDataset(data.Dataset, abc.ABC):

    def __init__(self, 
                 root: Union[Path, str], 
                 split: str,
                 frames_per_clip: int,
                 step_between_clips: int = 1,
                 frame_rate: int = None,
                 transform: Transform = None,
                 num_workers: int = 4,
                 extensions: Collection[str] = ('mp4', 'avi')) -> None:

        self.root = Path(root)
        self.split = split
        self.classes = sorted([o.stem for o in self.split_root.iterdir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = make_dataset(str(self.split_root), 
                                    self.class_to_idx, 
                                    extensions, 
                                    is_valid_file=None)
        self.videos_path = [x[0] for x in self.samples]

        self._video_clips = VideoClips(
            self.videos_path,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            None,
            num_workers=num_workers,
            _video_width=0,
            _video_height=0,
            _video_min_dimension=0,
            _audio_samples=0,
            _audio_channels=0)

        self.transform = transform

    @abc.abstractproperty
    def split_root(self) -> Path:
        raise NotImplemented
    
    @property
    def metadata(self) -> dict:
        return self.video_clips.metadata
    
    @property
    def video_clips(self) -> VideoClips:
        return self._video_clips

    def __len__(self) -> int:
        return self.video_clips.num_clips()
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, 
                                             torch.Tensor,
                                             int, dict]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, info


class Kinetics400(VideoDataset):

    def __init__(self, 
                 root: Union[Path, str], 
                 split: str,
                 frames_per_clip: int,
                 step_between_clips: int = 1,
                 frame_rate: int = None,
                 transform: Transform = None,
                 num_workers: int = 4,
                 extensions: Collection[str] = ('mp4', 'avi')) -> None:

        super(Kinetics400, self).__init__(
            root, split, frames_per_clip, step_between_clips, frame_rate,
            transform, num_workers, extensions)
    
    @property
    def split_root(self) -> Path:
        return self.root / self.split


class UCF101(VideoDataset):

    def __init__(self, 
                 root: Union[Path, str],
                 annotation_path: Union[Path, str],
                 split: str,
                 frames_per_clip: int,
                 fold: int = 1,
                 step_between_clips: int = 1,
                 frame_rate: int = None,
                 transform: Transform = None,
                 num_workers: int = 4,
                 extensions: Collection[str] = ('mp4', 'avi')) -> None:

        super(UCF101, self).__init__(
            root, split, frames_per_clip, step_between_clips, frame_rate,
            transform, num_workers, extensions)
        
        self.annotation_path = Path(annotation_path)
        
        # Filter the videoclips by annotations
        self.indices = self._select_fold(fold)
        self._video_clips = self._video_clips.subset(self.indices)

    @property
    def split_root(self) -> Path:
        return self.root

    def _select_fold(self, fold: int) -> Sequence[int]:
        assert self.split in {'train', 'test'}, \
            'split argument must be either "train" or "test"'

        name = f'{self.split}list{fold:02d}.txt'
        f = self.annotation_path / name

        video_files = f.read_text().split('\n')
        video_files = [o.strip().split()[0] for o in video_files if o]
        video_files = [str(self.split_root / o) for o in video_files]
        video_files = set(video_files)
        return [i for i in range(len(self.videos_path)) 
                if str(self.videos_path[i]) in video_files]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, 
                                             torch.Tensor,
                                             int, dict]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, info