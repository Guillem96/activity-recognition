import abc
import warnings
from pathlib import Path
from re import L
from typing import Collection
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets.video_utils import VideoClips

from ar.typing import PathLike
from ar.typing import Transform

_ClipDatasetSample = Tuple[torch.Tensor, torch.Tensor, int, dict]


class ClipLevelDataset(data.Dataset, abc.ABC):
    """Base class to provide a common interface among video datasets.

    The subclasses of this class are expected to work at clip level.

    Parameters
    ----------
    root: PathLike
        Path to the folder containing the video clips
    split: str
        Data split that you are using: train, valid, train01, etc.
    frames_per_clip: int
        Number of frames per sampled clip
    step_between_clip: int, defaults 1
        Stride between sampled clips.
    frame_rate: Optional[int], defaults to None
        Resample the sampled clips at the given frame_rate. If left to None
        the clips are not resampled
    transform: Optional[Transform], defaults None
        Callables that receive a tensor and returns a modified tensor. This
        parameter is useful for data augmentation.
    num_workers: int, defaults 4
        Number of processes to index the video clips
    extensions: Collection[str], defaults ('mp4', 'avi')
        Files extensions of the videos
    """

    def __init__(
        self,
        root: PathLike,
        split: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        transform: Optional[Transform] = None,
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

        self._classes: Optional[List[str]] = None
        self._class_2_idx: Optional[Mapping[str, int]] = None

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

    @property
    def n_videos(self):
        """Get the number of videos referenced by the dataset."""
        return len(self.paths)

    @property
    def split_root(self) -> Path:
        """Get the directory of the videos belonging to a split.

        Some datasets like UCF-101 do not use these method since the
        splits are defined in a text file.

        Examples
        --------
        def split_root(self) -> Path:
            return self.root / self.split

        Returns
        -------
        Path
            Path to the folder containing the videos for a given split
        """
        return self.root

    @abc.abstractproperty
    def paths(self) -> Sequence[Path]:
        """Get a list of all videos of the dataset.

        Recurse in the self.root directory may be a good idea.

        Returns
        -------
        Sequence[Path]
            List of all videos paths. 
        """

    @abc.abstractproperty
    def labels(self) -> Sequence[str]:
        """List of all video labels.

        Please note that the position i of the list returned here must be the
        label of the self.paths[i]

        Returns
        -------
        Sequence[str]
            List of labels. The label at position i matches the class of the
            path at position i
        """

    @property
    def classes(self) -> List[str]:
        """Returns a list of the dataset classes.

        It is computed by taking the unique values returned by th self.labels
        property.

        Returns
        -------
        List[str]
            List of dataset classes
        """
        if self._classes:
            return self._classes
        self._classes = sorted(list(set(self.labels)))
        return self._classes

    @property
    def class_2_idx(self) -> Mapping[str, int]:
        """Get the dictionary to encode a label

        Returns
        -------
        Mapping[str, int]
            Class string to index mapping.
        """
        if self._class_2_idx:
            return self._class_2_idx
        self._class_2_idx = {c: i for i, c in enumerate(self.classes)}
        return self._class_2_idx

    @property
    def metadata(self) -> dict:
        """VideoClips metadata

        Delegates to VideoClips property.

        Returns
        -------
        dict
            VideoClips metadata
        """
        return self.video_clips.metadata

    def __len__(self) -> int:
        """Magic method that computes the length of the dataset.

        This is required by PyTorch

        Returns
        -------
        int
            Length
        """
        return self.video_clips.num_clips()

    def __getitem__(self, idx: int) -> _ClipDatasetSample:
        """Fetch an indexed sample.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        _ClipDatasetSample
            Dataset sample being a tuple of four items containing the video,
            the audio, the video information and the label.
        """
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.class_2_idx[self.labels[video_idx]]

        if self.transform is not None:
            video = self.transform(video)

        return video, audio, label, info


class VideoLevelDataset(data.Dataset, abc.ABC):
    """Base class to provide a common interface to iterate over video files.

    VideoLevelDatasets do not load the whole video in RAM memory. Instead it
    computes the paths and labels of the videos and the `__getitem__` method
    returns the video path along the corresponding encoded label.

    To process the video once sampled from the dataset we recommend the builtin
    VideoFramesIterator (`ar.io.VideoFramesIterator`).

    This datasets are iterable:

    .. code-block:: python

        ds = VideoLevelKinetics('root_path/', 'train')

        # Iterate flavour
        for path, label in ds:
            print(path, label)

        # Or indexed
        for i in range(len(dataset)):
            path, label = ds[i]
            print(path, label)

    Parameters
    ----------
    video_paths: All video paths
        Paths to videos
    labels: Sequence[str]
        Label corresponding to each path. The position i of this list 
        is the label of the video at the position i of the `video_paths` 
        argument.
    """

    def __init__(self, video_paths: Sequence[PathLike],
                 labels: Sequence[str]) -> None:

        self.video_paths = [str(o) for o in video_paths]
        self.labels = labels
        self.classes = sorted(list(set(labels)))
        self.class_2_idx = {c: i for i, c in enumerate(self.classes)}
        self.labels_ids = [self.class_2_idx[o] for o in self.labels]

        self._pos = 0

    def __iter__(self) -> "VideoLevelDataset":
        return self

    def __next__(self) -> Tuple[PathLike, int]:
        if self._pos == len(self):
            raise StopIteration
        sample = self[self._pos]
        self._pos += 1
        return sample

    def __len__(self) -> int:
        """Gets the dataset length."""
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> Tuple[PathLike, int]:
        """Fetches a sample from the dataset
        
        Parameters
        ----------
        idx: int
            Sample unique index
        
        Returns
        -------
        Tuple[PathLike, int]
            Path pointing to the video file and the encoded label.
        """

        return self.video_paths[idx], self.class_2_idx[self.labels[idx]]
