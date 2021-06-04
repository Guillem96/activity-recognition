import itertools
import math
from pathlib import Path
from typing import Optional
from typing import Tuple

import torch
import torchvision

from ar.typing import PathLike
from ar.typing import Transform


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
    batch_size: int
        Number of frames to retrieve after each iteration
    skip_frames: int, defaults 1
        Skip frames to process the video faster but with less precision
    transforms: Optional[Transform], defaults None
        Video transformations
    """

    def __init__(self,
                 video_path: PathLike,
                 batch_size: int,
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
             from_sec: int,
             to_sec: int,
             do_skip_frames: bool = False,
             do_transform: bool = False) -> torch.Tensor:
        """Take the frames between two timestamps in seconds.

        Parameters
        ----------
        from_sec: int
            Init of the interval
        to_sec: int
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
