import abc
from pathlib import Path
from typing import Tuple, Collection, Union, Sequence

import torch
import torch.utils.data as data

import torchvision
from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

from ar.typing import Transform


class ClipLevelDataset(data.Dataset, abc.ABC):

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
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.frame_rate = frame_rate
        self.transform = transform
        self._num_workers = num_workers
        self._extensions = extensions

    def _build_video_clips(self, 
                           paths: Collection[Union[str, Path]]) -> VideoClips:
        return VideoClips(
            paths,
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
    
    @property
    def metadata(self) -> dict:
        return self.video_clips.metadata
    
    @abc.abstractproperty
    def video_clips(self) -> VideoClips:
        raise NotImplementedError()
    
    @abc.abstractproperty
    def classes(self) -> Sequence[str]:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.video_clips.num_clips()
    
    @abc.abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, 
                                             torch.Tensor,
                                             int, dict]:
        raise NotImplementedError()


class Kinetics400(ClipLevelDataset):

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

        self._classes = sorted([o.stem for o in self.split_root.iterdir()])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.samples = make_dataset(str(self.split_root), 
                                    self.class_to_idx, 
                                    extensions, 
                                    is_valid_file=None)
        self.videos_path = [x[0] for x in self.samples]

        self._video_clips = self._build_video_clips(self.videos_path)

    @property
    def video_clips(self) -> VideoClips:
        return self._video_clips

    @property
    def classes(self) -> Sequence[str]:
        return self._classes

    @property
    def split_root(self) -> Path:
        return self.root / self.split

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, 
                                             torch.Tensor,
                                             int, dict]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[video_idx][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, info


class UCF101(ClipLevelDataset):

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

        assert self.split in {'train', 'test'}, \
            'split argument must be either "train" or "test"'

        self.annotation_path = Path(annotation_path)
        self.videos_path = ucf_select_fold(self.split_root, 
                                           self.annotation_path, 
                                           self.split, fold)
        self.videos_path = [o for o in self.videos_path 
                            if o.suffix[1:] in extensions]
        
        # Get the video labels in str format
        self.labels = [o.parent.stem for o in self.videos_path]
        self.videos_path = list(map(str, self.videos_path))
        
        # Unique elements in the whole labels collection are the classes
        self._classes = sorted(list(set(self.labels)))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self._video_clips = self._build_video_clips(self.videos_path)

        self.transform = transform

    @property
    def video_clips(self) -> VideoClips:
        return self._video_clips

    @property
    def classes(self) -> Sequence[str]:
        return self._classes

    @property
    def split_root(self) -> Path:
        return self.root

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, 
                                             torch.Tensor,
                                             int, dict]:
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.class_to_idx[self.labels[video_idx]]

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
        video, audio, info = torchvision.io.read_video(
            self.video_paths[idx])
        
        video = self.resample_video(video, info['video_fps'])

        if self.transform is not None:
            video = self.transform(video)
        
        return video, audio, self.class_2_idx[self.labels[idx]]


class VideoLevelUCF101(VideoLevelDataset):

    def __init__(self, 
                 root: Union[Path, str],
                 annotation_path: Union[Path, str],
                 split: str,
                 fold: int = 1,
                 frame_rate: int = None,
                 transform: Transform = None):
        
        assert split in {'train', 'test'}, \
            'split argument must be either "train" or "test"'

        video_paths = ucf_select_fold(Path(root) / split, 
                                      Path(annotation_path),
                                      split, fold)
        labels = [o.parent.stem for o in video_paths]
        super(VideoLevelUCF101, self).__init__(
            video_paths, labels, frame_rate, transform)


class VideoLevelKinetics(VideoLevelDataset):

    def __init__(self, 
                 root: Union[Path, str],
                 split: str,
                 frame_rate: int = None,
                 transform: Transform = None):
        
        assert split in {'train', 'test', 'valid'}, \
            'split argument must be either "train", "valid" or "test"'
        
        path = Path(root, split)
        classes = sorted([o.stem for o in path.iterdir()])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        samples = make_dataset(str(path), class_to_idx, 
                               ('mp4', 'avi'), is_valid_file=None)

        video_paths = [o[0] for o in samples]
        labels = [classes[i[1]] for i in samples]

        super(VideoLevelKinetics, self).__init__(
            video_paths, labels, frame_rate, transform)


def ucf_select_fold(base_path: Path,
                    annotation_path: Path,
                    split: str,
                    fold: int) -> Sequence[Path]:
    name = f'{split}list{fold:02d}.txt'
    f = annotation_path / name

    video_files = f.read_text().split('\n')
    video_files = [o.strip().split()[0] for o in video_files if o]
    return list(set([base_path / o for o in video_files]))
