from pathlib import Path
from typing import Union

from torchvision.datasets.folder import make_dataset

from ar.data.datasets.base import VideoLevelDataset
from ar.data.datasets.utils import ucf_select_fold
from ar.typing import PathLike
from ar.typing import Transform


class VideoLevelUCF101(VideoLevelDataset):
    """Video level dataset for Kinetics dataset format

     This datasets are iterable:

    .. code-block:: python

        ds = VideoLevelUCF101('root_path/', 'annots/', 'train')

        # Iterate flavour
        for path, label in ds:
            print(path, label)

        # Or indexed
        for i in range(len(dataset)):
            path, label = ds[i]
            print(path, label)

    Parameters
    ----------
    Parameters
    ----------
    root: PathLike
        Path to the folder containing the video clips
    annotation_path: PathLike
        Path to the folder containing the annotations
    split: str
        Data split that you are using: train, or test.
    """

    def __init__(self,
                 root: PathLike,
                 annotation_path: PathLike,
                 split: str,
                 fold: int = 1) -> None:

        assert split in {'train', 'test'}, \
            'split argument must be either "train" or "test"'

        video_paths = ucf_select_fold(root, annotation_path, split, fold)
        labels = [o.parent.stem for o in video_paths]
        super(VideoLevelUCF101, self).__init__(video_paths, labels)


class VideoLevelKinetics(VideoLevelDataset):
    """Video level dataset for Kinetics dataset format

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
    Parameters
    ----------
    root: PathLike
        Path to the folder containing the video clips
    split: str
        Data split that you are using: train, valid or test.
    """

    def __init__(self, root: PathLike, split: str) -> None:

        assert split in {'train', 'test', 'valid'}, \
            'split argument must be either "train", "valid" or "test"'

        path = Path(root, split)
        classes = sorted([o.stem for o in path.iterdir()])
        class_to_idx = {c: i for i, c in enumerate(classes)}
        samples = make_dataset(str(path),
                               class_to_idx, ('mp4', 'avi'),
                               is_valid_file=None)

        video_paths = [o[0] for o in samples]
        labels = [classes[i[1]] for i in samples]

        super(VideoLevelKinetics, self).__init__(video_paths, labels)
