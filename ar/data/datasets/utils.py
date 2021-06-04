from pathlib import Path
from typing import Sequence

from ar.typing import PathLike


def ucf_select_fold(base_path: PathLike, annotation_path: PathLike, split: str,
                    fold: int) -> Sequence[Path]:
    """Given a root and a UCF annotation file filters the paths.

    Parameters
    ----------
    base_path: PathLike
        Root path of the videos
    annotation_path: PathLike
        Directory containing the annotations
    split: str
        Annotation set
    fold: int
        K fold of the annotation set

    Returns
    -------
    Sequence[Path]
        Paths belonging to the annotation set and fold
    """
    base_path = Path(base_path)
    annotation_path = Path(annotation_path)

    name = f'{split}list{fold:02d}.txt'
    f = annotation_path / name

    video_files = f.read_text().split('\n')
    video_files = [o.strip().split()[0] for o in video_files if o]
    return list(set([base_path / o for o in video_files]))
