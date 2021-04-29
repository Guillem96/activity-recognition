import math
import itertools
from pathlib import Path
from typing import Optional

import torch
import torchvision

from ar.typing import Transform


class VideoFramesIterator(object):
    def __init__(self,
                 video_path: Path,
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
        return self.metadata['video']['fps'][0]

    @property
    def video_duration(self) -> float:
        return self.metadata['video']['duration'][0]

    @property
    def total_frames(self) -> int:
        return int(self.video_fps * self.video_duration)

    def take(self,
             from_sec: int,
             to_sec: int,
             do_skip_frames: bool = False,
             do_transform: bool = False) -> torch.Tensor:
        video_it = self._video_reader.seek(from_sec)
        frames = [
            f['data'] for i, f in enumerate(
                itertools.takewhile(lambda x: x['pts'] < to_sec, video_it))
            if not do_skip_frames or (i % self.skip_frames == 0)
        ]

        frames = torch.stack(frames).permute(0, 2, 3, 1)
        if do_transform:
            return self.tranforms(frames)
        return frames

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
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
        return math.ceil(
            (self.total_frames / self.skip_frames) / self.batch_size)
