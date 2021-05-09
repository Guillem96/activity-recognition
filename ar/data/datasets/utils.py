from pathlib import Path
from typing import Sequence
from typing import Union


def ucf_select_fold(base_path: Union[Path, str], annotation_path: Union[Path,
                                                                        str],
                    split: str, fold: int) -> Sequence[Path]:
    base_path = Path(base_path)
    annotation_path = Path(annotation_path)

    name = f'{split}list{fold:02d}.txt'
    f = annotation_path / name

    video_files = f.read_text().split('\n')
    video_files = [o.strip().split()[0] for o in video_files if o]
    return list(set([base_path / o for o in video_files]))


class IdentityMapping(dict):

    def __missing__(self, key):
        return key
