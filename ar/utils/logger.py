from collections import deque
from datetime import datetime
from math import log
from pathlib import Path
from typing import Any
from typing import Deque
from typing import Dict
from typing import Mapping
from typing import Tuple
from typing import Union

import torch
import tqdm.auto as tqdm

from ar.data import ClipLevelDataset
from ar.transforms import imagenet_stats
from ar.transforms.functional import video_unnormalize
from ar.typing import Number
from ar.typing import PathLike
from ar.typing import TensorBoard


class LogValue(object):
    """
    Utility class to simplify the logging of a value. Keep tracks of the 
    last `window_size` values and can perform aggregations such the mean
    or the median

    Parameters
    ----------
    name: str
        Value name
    window_size: int
        How many values should we keep track
    
    """

    def __init__(self, name: str, window_size: int):
        self.name = name
        self.window_size = window_size
        self.logs: Deque[Number] = deque(maxlen=window_size)

    def __call__(self, new_value: Number) -> None:
        """Append a new value to the tracking list"""
        self.logs.append(new_value)

    def __str__(self) -> str:
        return f'{self.name}: {self.mean.item():.6f}'

    def reset(self) -> None:
        """Remove the tracking queue"""
        self.logs.clear()

    @property
    def mean(self) -> torch.Tensor:
        if self.window_size == 1:
            return torch.tensor(self.logs[0]).float()

        logs = torch.as_tensor(self.logs).float()
        return torch.mean(logs)

    @property
    def median(self) -> torch.Tensor:
        if self.window_size == 1:
            return torch.tensor(self.logs[0]).float()

        logs = torch.as_tensor(self.logs).float()
        return torch.median(logs)


class ValuesLogger(object):
    """
    Utility class to simplify the activity of logging LogValues. This object
    has the capability of updating multiple `LogValues` at the same time and
    log them every ``print_freq`` steps. It also keeps track of the time
    elapsed between updates.

    Parameters
    ----------
    *values: sequence of LogValue
        LogValues to manage
    total_steps: int
        Total logger lifetime
    header: str, default ''
        Text to add befor the values logs
    """

    def __init__(self,
                 *values: LogValue,
                 total_steps: int,
                 header: str = '') -> None:
        self.values = {v.name: v for v in values}
        self.t = tqdm.trange(total_steps, desc=header, leave=True)

    def __call__(self, **kwargs: Number) -> None:

        # Update the values
        for k, v in kwargs.items():
            self.values[k](v)

        self.t.set_postfix(self.as_dict())
        self.t.update()
        self.t.refresh()

    def reset(self) -> None:
        """Reset all values and set steps to 0"""
        for v in self.values.values():
            v.reset()

    def as_dict(self) -> Dict[str, float]:
        return {k: v.mean.item() for k, v in self.values.items()}


def build_summary_writter(log_dir: PathLike) -> TensorBoard:
    """
    Builds a hierarchical log directory structure for ease of experiments 
    tracking

    Parameters
    ----------
    log_dir: PathLike
        Base log directory. Given a log_dir this method adds a unique identifier
        as a postfix
    """
    log_dir = Path(log_dir)
    current_time = datetime.now().strftime('%b%d_%H_%M_%S')
    log_dir = log_dir / f'experiment_{current_time}'
    return torch.utils.tensorboard.SummaryWriter(str(log_dir), flush_secs=20)


def log_random_videos(ds: ClipLevelDataset,
                      writer: TensorBoard,
                      samples: int = 4,
                      unnormalize_videos: bool = False,
                      video_format: str = 'THWC') -> None:
    """
    Log `samples` clips to tensorboard

    Parameters
    ----------
    ds: ar.data.ClipLevelDataset
        Dataset to get the random samples
    writer: TensorBoard
        TensorBoard summary writer
    samples: int, default 4
        Pick n samples of the dataset
    unnormalize_videos: bool, default False
        Wether to unnormalize the videos or not
    video_format: str, default "THWC"
        Shape of the video returned by the dataset. 
            - T timesteps
            - H height
            - W width
            - C color channels 
    """
    for c in 'THWC':
        if c not in video_format:
            raise ValueError(f'video_format does not contain {c}')

    for c in video_format:
        if c not in 'THWC':
            raise ValueError(f'Invalid character {c} for video_format')

    if writer is None:
        return

    indices = torch.randint(high=len(ds), size=(samples,)).tolist()
    videos = []
    labels = []
    for i in indices:
        video, _, label, _ = ds[i]
        if unnormalize_videos:
            video = video.permute(video_format.index('C'),
                                  video_format.index('T'),
                                  video_format.index('H'),
                                  video_format.index('W'))
            video = video_unnormalize(video, **imagenet_stats)

        video = video.permute(video_format.index('T'), video_format.index('C'),
                              video_format.index('H'), video_format.index('W'))

        label_name = ds.classes[int(label)]
        videos.append(video)
        labels.append(label_name)

    videos = torch.stack(videos)
    tag = ' - '.join(labels)

    writer.add_video(tag, videos)
