from pathlib import Path
from typing import Union

from torchvision.datasets.folder import make_dataset

from ar.data.datasets.base import VideoLevelDataset
from ar.data.datasets.utils import ucf_select_fold
from ar.typing import Transform


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

        video_paths = ucf_select_fold(root, annotation_path, split, fold)
        labels = [o.parent.stem for o in video_paths]
        super(VideoLevelUCF101, self).__init__(video_paths, labels, frame_rate,
                                               transform)


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
        samples = make_dataset(str(path),
                               class_to_idx, ('mp4', 'avi'),
                               is_valid_file=None)

        video_paths = [o[0] for o in samples]
        labels = [classes[i[1]] for i in samples]

        super(VideoLevelKinetics, self).__init__(video_paths, labels,
                                                 frame_rate, transform)

