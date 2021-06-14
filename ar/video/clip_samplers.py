from typing import List
from typing import Tuple

import torch

from ar.io import VideoFramesIterator


def uniform_sampling(video: VideoFramesIterator,
                     clips_len: int,
                     n_clips: int = 12,
                     overlap: bool = True) -> List[torch.Tensor]:
    """
    Uniformly samples `n_clips` from a video. The clips samples by default can 
    be overlapping.

    Parameters
    ----------
    video: VideoFramesIterator
        VideoFramesIterator to take the clips from.
    clips_len: int
        Length of the sampled clips in frames
    n_clips: int, default 12
        Number of clips to sample
    overlap: bool, default True
        If set to False the sampled clips won't be overlapping

    Returns
    -------
    List[torch.Tensor]
        List of tensors containing `n_clips` of shape according to video_fmt, 
        where T will be `clips_len`
    """
    clips_in_sec = clips_len / (video.video_fps / video.skip_frames)

    if overlap:
        start_secs = torch.rand(size=(n_clips,))
        start_secs = start_secs * (video.video_duration - clips_in_sec - 1)
        start_secs = start_secs.tolist()
    else:
        # If it is no possible to sample n_clips reduce it progressively until
        # it is possible
        clips_duration = clips_in_sec * n_clips
        while clips_duration > video.video_duration:
            n_clips -= 1
            clips_duration = clips_in_sec * n_clips

        possible_start_secs = torch.arange(0,
                                           video.video_duration - clips_in_sec,
                                           clips_in_sec,
                                           dtype=torch.float32)
        choices = torch.randperm(possible_start_secs.size(0))[:n_clips]
        start_secs = possible_start_secs[choices]

    return [
        video.take(ss, ss + clips_in_sec, do_transform=True, limit=clips_len)
        for ss in start_secs
    ]


def lrcn_sampling(video: VideoFramesIterator,
                  clips_len: int,
                  n_clips: int = 16,
                  stride: int = 8) -> List[torch.Tensor]:
    """
    Sample clips as described in LRCN paper. 
    Citation: At test time, we extract 16 frame clips with a stride of 8 frames
    from each video.

    If n_clips is larger than the total clips available in video the resulting
    list will have at most the clips available in the video.

    Parameters
    ----------
    VideoFramesIterator: torch.Tensor
        VideoFramesIterator to take the clips from.
    clips_len: int
        Length of the sampled clips in frames
    n_clips: int, default 16
        Number of clips to sample
    stride: int, default 8
        Distance in frames 

    Returns
    -------
    List[torch.Tensor]
        List of tensors containing `n_clips` of shape equal to the returned by
        the VideoFramesIterator.
    """
    clips_in_sec = clips_len / (video.video_fps / video.skip_frames)
    stride_in_sec = stride / (video.video_fps / video.skip_frames)
    samples_duration = (clips_in_sec + stride_in_sec) * n_clips

    start_sec = torch.rand((1,)).item()
    start_sec = start_sec * video.video_duration - samples_duration
    clips_start_idx = torch.arange(start_sec, video.video_duration,
                                   stride_in_sec + clips_in_sec)[:n_clips]

    return [
        video.take(ss, ss + clips_in_sec, do_transform=True, limit=clips_len)
        for ss in clips_start_idx
    ]


def FstCN_sampling(video: VideoFramesIterator,
                   clips_len: int,
                   n_clips: int = 16,
                   n_crops: int = 4,
                   crops_size: Tuple[int, int] = (112, 112),
                   overlap: bool = False) -> List[List[torch.Tensor]]:
    """
    Uniformly sampling `n_clips` from a given video, and for each clip, we
    randomly sample `n_crops` different crops.

    Parameters
    ----------
    video: VideoFramesIterator
        VideoFramesIterator to take the clips from. It is mandatory that the
        VideoFramesIterator returns a tensor of shape 
        [FRAMES, CHANNELS, HEIGHT, WIDTH]
    clips_len: int
        Length of the sampled clips in frames
    n_clips: int, default 16
        Number of clips to sample
    n_crops: int, default 4
        Number of crops to extract for each video clip
    crops_size: Tuple[int, int], default (112, 112)
        Size of the generated crops
    overlap: bool, default True
        When sampling clips, can clips be overlapping

    Returns
    -------
    List[torch.Tensor]
        Each list item contains a list of clips cropped at different points.
        The tensors have the shape [FRAMES, CHANNELS, HEIGHT, WIDTH].
        The list will contain n_clips * n_crops * 2 clips
    """
    import ar.transforms as VT

    crop_fn = VT.VideoRandomCrop(crops_size)

    clips = uniform_sampling(video=video,
                             clips_len=clips_len,
                             n_clips=n_clips,
                             overlap=overlap)

    results = []
    for clip in clips:
        cropped_clips = [crop_fn(clip) for _ in range(n_crops)]
        flipped_clips = [o.flip(dims=-1) for o in cropped_clips]
        results.extend(cropped_clips + flipped_clips)

    return results
