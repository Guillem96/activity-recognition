from pathlib import Path
from typing import Collection
from typing import Hashable
from typing import Optional
from typing import Sequence
from typing import Union

from torchvision.datasets.folder import make_dataset

from ar.data.datasets.base import ClipLevelDataset
from ar.data.datasets.utils import ucf_select_fold
from ar.typing import Transform


class Kinetics400(ClipLevelDataset):
    """Clip level dataset for kinetics dataset format.

    Kinetics format is the same as ImageNet dataset, but instead of images.

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
        root: Union[Path, str],
        split: str,
        frames_per_clip: int,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        transform: Optional[Transform] = None,
        num_workers: int = 4,
        extensions: Collection[str] = ('mp4', 'avi')) -> None:

        super(Kinetics400,
              self).__init__(root, split, frames_per_clip, step_between_clips,
                             frame_rate, transform, num_workers, extensions)

        self.samples = make_dataset(str(self.split_root),
                                    _IdentityMapping(),
                                    extensions,
                                    is_valid_file=None)
        self._videos_path = [x[0] for x in self.samples]
        self._labels = [x[1] for x in self.samples]

        self.video_clips = self._build_video_clips()

    @property
    def paths(self) -> Sequence[Path]:
        """Get a list of all videos of the dataset.

        Recurse in the self.root directory may be a good idea.

        Returns
        -------
        Sequence[Path]
            List of all videos paths. 
        """
        return self._videos_path

    @property
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
        return self._labels

    @property
    def split_root(self) -> Path:
        """Get the directory of the videos belonging to a split.

        If split="train" this property returns self.root / "train"

        Returns
        -------
        Path
            Path to the folder containing the videos for a given split
        """
        return self.root / self.split


class UCF101(ClipLevelDataset):
    """Clip level dataset for UCF-101 dataset format.

    The UCF-101 is composed of two main directories:
    
    One containing the a directory for each class, and these subdirectories
    contain all the videos of the corresponding class.

    The other directory contains a set of text files with the annotations.

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
        root: Union[Path, str],
        annotation_path: Union[Path, str],
        split: str,
        frames_per_clip: int,
        fold: int = 1,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        transform: Optional[Transform] = None,
        num_workers: int = 4,
        extensions: Collection[str] = ('mp4', 'avi')) -> None:

        super(UCF101,
              self).__init__(root, split, frames_per_clip, step_between_clips,
                             frame_rate, transform, num_workers, extensions)

        assert self.split in {'train', 'test'}, \
            'split argument must be either "train" or "test"'

        self.annotation_path = Path(annotation_path)
        videos_path = ucf_select_fold(self.split_root, self.annotation_path,
                                      self.split, fold)
        self._videos_path = [
            o for o in videos_path if o.suffix[1:] in extensions
        ]

        # Get the video labels in str format
        self._labels = [o.parent.name for o in videos_path]

        self.video_clips = self._build_video_clips()

    @property
    def paths(self) -> Sequence[Path]:
        """Get a list of all videos of the dataset.

        Recurse in the self.root directory may be a good idea.

        Returns
        -------
        Sequence[Path]
            List of all videos paths. 
        """
        return self._videos_path

    @property
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
        return self._labels


class _IdentityMapping(dict):

    def __missing__(self, key: Hashable) -> Hashable:
        return key
