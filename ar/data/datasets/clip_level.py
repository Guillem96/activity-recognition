from pathlib import Path
from typing import Collection
from typing import Sequence
from typing import Union

from torchvision.datasets.folder import make_dataset
from torchvision.datasets.video_utils import VideoClips

from ar.data.datasets.base import ClipLevelDataset
from ar.data.datasets.utils import IdentityMapping
from ar.data.datasets.utils import ucf_select_fold
from ar.typing import Transform


class Kinetics400(ClipLevelDataset):
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

        super(Kinetics400,
              self).__init__(root, split, frames_per_clip, step_between_clips,
                             frame_rate, transform, num_workers, extensions)

        self.samples = make_dataset(str(self.split_root),
                                    IdentityMapping(),
                                    extensions,
                                    is_valid_file=None)
        self._videos_path = [x[0] for x in self.samples]
        self._labels = [x[1] for x in self.samples]
        self._video_clips = self._build_video_clips()

    @property
    def paths(self) -> Sequence[Path]:
        return self._videos_path

    @property
    def labels(self) -> Sequence[str]:
        return self._labels

    @property
    def video_clips(self) -> VideoClips:
        return self._video_clips

    @property
    def split_root(self) -> Path:
        return self.root / self.split


class UCF101(ClipLevelDataset):
    def __init__(
        self,
        root: Union[Path, str],
        annotation_path: Union[Path, str],
        split: str,
        frames_per_clip: int,
        fold: int = 1,
        step_between_clips: int = 1,
        frame_rate: int = None,
        transform: Transform = None,
        num_workers: int = 4,
        extensions: Collection[str] = ('mp4', 'avi')
    ) -> None:

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

        # Unique elements in the whole labels collection are the classes
        self._video_clips = self._build_video_clips()

    @property
    def video_clips(self) -> VideoClips:
        return self._video_clips

    @property
    def classes(self) -> Sequence[str]:
        return self._classes

    @property
    def split_root(self) -> Path:
        return self.root
