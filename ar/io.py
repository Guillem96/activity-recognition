import itertools
import math
from pathlib import Path
from typing import Any, Optional
from typing import Tuple

import torch
import torchvision

from ar.typing import PathLike
from ar.typing import Transform

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.figure import Figure


class VideoFramesIterator(object):
    """Iterates over the given video.

    With this object you can iterate over a video with a batched approach. 
    This means that you can progressively take `batch_size` frames at every
    iteration.

    This object is useful to safe memory when processing a video.

    Parameters
    ----------
    video_path: PathLike
        Video filesystem path
    batch_size: int, defaults 1
        Number of frames to retrieve after each iteration
    skip_frames: int, defaults 1
        Skip frames to process the video faster but with less precision
    transforms: Optional[Transform], defaults None
        Video transformations
    """

    def __init__(self,
                 video_path: PathLike,
                 batch_size: int = 1,
                 skip_frames: int = 1,
                 transforms: Optional[Transform] = None) -> None:
        self._video_reader = torchvision.io.VideoReader(str(video_path))
        self.current_frame = 0
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.tranforms = transforms or (lambda x: x)
        self._is_it_end = False
        self.metadata = self._video_reader.get_metadata()

    @property
    def video_fps(self) -> float:
        """Get video FPS

        Returns
        -------
        float
            Video's FPS
        """
        return self.metadata['video']['fps'][0]

    @property
    def video_duration(self) -> float:
        """Get the video duration in seconds

        Returns
        -------
        float
            Video duration in seconds
        """
        return self.metadata['video']['duration'][0]

    @property
    def total_frames(self) -> int:
        """Get the total number of frames.

        Returns
        -------
        int
            Number of frames that the video is composed of.
        """
        return int(self.video_fps * self.video_duration)

    def take(self,
             from_sec: float,
             to_sec: float,
             do_skip_frames: bool = False,
             do_transform: bool = False) -> torch.Tensor:
        """Take the frames between two timestamps in seconds.

        Parameters
        ----------
        from_sec: float
            Init of the interval
        to_sec: float
            End of the interval
        do_skip_frames: bool, defaults Fale
            Skip the frames specified in the __init__?
        do_transform: bool, defaults False
            Apply the given transformations

        Returns
        -------
        torch.Tensor
            Tensor containing the stacked frames. The shape before being fed to
            the self.transform callable is [FRAMES, HEIGHT, WIDTH, CHANNELS]
        """
        video_it = self._video_reader.seek(from_sec)
        frames = [
            f['data']
            for i, f in enumerate(
                itertools.takewhile(lambda x: x['pts'] < to_sec, video_it))
            if not do_skip_frames or (i % self.skip_frames == 0)
        ]

        frames = torch.stack(frames).permute(0, 2, 3, 1)
        if do_transform:
            return self.tranforms(frames)
        return frames

    def __iter__(self) -> 'VideoFramesIterator':
        """Python iterator interface"""
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get next frames batch

        Examples
        --------
        >>> vit = VideoFramesIterator("path", batch_size=4, skip_frames=2)
        >>> frames, idx = next(vit)
        >>> frames.size
        torch.Size([4, 112, 168, 3])
        >>> idx  # Selected frames are:
        [0, 2, 4, 6]
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of two tensors, the first tensor contains the stacked frames. 
            The shape before being fed to the self.transform callable is 
            [FRAMES, HEIGHT, WIDTH, CHANNELS], the second tensor contains the 
            frame indices
        """
        if self._is_it_end:
            raise StopIteration

        frames = []
        frames_idx = []
        start = self.current_frame
        end = min(start + self.batch_size * self.skip_frames, self.total_frames)

        for _ in range(start, int(end)):
            try:
                frame = next(self._video_reader)
            except StopIteration:
                self._is_it_end = True
                break

            if self.current_frame % self.skip_frames == 0:
                frames.append(frame['data'])
                frames_idx.append(self.current_frame)

            self.current_frame += 1

        if not frames:
            self._is_it_end = True
            raise StopIteration

        # (FRAMES, CHANNELS, HEIGHT, WIDTH) to (FRAMES, HEIGHT, WIDTH, CHANNELS)
        video_clip = torch.stack(frames).permute(0, 2, 3, 1)
        video_clip = self.tranforms(video_clip)
        frames_idx = torch.as_tensor(frames_idx, dtype=torch.long)

        return frames_idx, video_clip

    def __len__(self) -> int:
        """Length of the iterator"""
        return math.ceil(
            (self.total_frames / self.skip_frames) / self.batch_size)


def plot_video(video: torch.Tensor,
               fig: Optional[Figure] = None,
               title: str = '') -> animation.Animation:
    """Plots a video using matplotlib.

    Parameters
    ----------
    video: torch.Tensor
        Of shape [FRAMES, HEIGHT, WIDTH, CHANNELS]
    fig: Optional[Figure]
        Matplotlib figure used to generate the animation
    title: str, defaults ''
        Title to display at the generated matplotlib figure

    Returns
    -------
    animation.Animation
        Matplotlib animation, you can plot it with plt.show()
    """
    fig = fig or plt.figure()
    plt.title(title)
    plt.axis('off')
    im = plt.imshow(video[0, :, :, :])

    def init() -> None:
        im.set_data(video[0, :, :, :])

    def animate(i: int) -> Any:
        im.set_data(video[i, :, :, :])
        return im

    return animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init,
                                   frames=video.shape[0],
                                   interval=50)


def plot_video_grid(
    clips: torch.Tensor,
    cols: int,
    rows: int,
    title: str = '',
    figsize: Tuple[int, int] = (15, 5)) -> animation.Animation:
    """Plot a grid of animated videos.

    Parameters
    ----------
    clips: torch.Tensor
        Tensor of shape [N, FRAMES, H, W, C]
    cols: int
        Grid columns
    rows: int
        Grid rows
    title: str, defaults ''
        Plot title, directly feed to plt.title
    figsize: Tuple[int, int], defaults (15, 5)
        Size of the matplotlib figure

    Returns
    -------
    animation.Animation
        Generated matplotlib animation
    """
    def animate(i: int) -> None:
        for j, ax in enumerate(axes):
            ax.set_data(clips[j][i])

    def init_fn() -> None:
        for j, ax in enumerate(axes):
            ax.set_data(clips[j][0])

    fig = plt.figure(figsize=figsize)
    plt.suptitle(title)
    axes = []
    for i, clip in enumerate(clips, start=1):
        plt.subplot(rows, cols, i)
        plt.axis('off')
        im = plt.imshow(clip.numpy()[0])
        axes.append(im)

    return animation.FuncAnimation(fig,
                                   animate,
                                   init_func=init_fn,
                                   frames=clips[0].size(0),
                                   interval=50)
