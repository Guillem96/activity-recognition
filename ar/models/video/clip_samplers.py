from typing import List

import torch


def uniform_sampling(video: torch.Tensor, 
                     clips_len: int, 
                     n_clips: int = 10,
                     overlap: bool = True) -> List[torch.Tensor]:
    """
    Uniformly samples `n_clips` from a video. The clips samples by default can 
    be overlapping.

    Parameters
    ----------
    video: torch.Tensor
        Tensor of shape [T, H, W, C] or [T, C, H, W]
    clips_len: int
        Length of the sampled clips in frames
    n_clips: int, default 3
        Number of clips to sample
    overlap: bool, default True
        If set to False the sampled clips won't be overlapping

    Returns
    -------
    List[torch.Tensor]
        List of tensors containing `n_clips` of shape [clips_len, H, W, C]
    """
    n_frames = video.size(0)
    if overlap:
        indices = torch.randint(high=n_frames - clips_len, size=(n_clips,))
        indices = indices.tolist()
    else:
        possible_indices = torch.arange(0, n_frames, clips_len)[:-1]
        choices = torch.randint(high=possible_indices.size(0), size=(n_clips,))
        indices = possible_indices[choices].tolist()
    
    clips = [video[i: i + clips_len] for i in indices]
    return clips


def lrcn_sampling(video: torch.Tensor, 
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
    video: torch.Tensor
        Tensor of shape [T, H, W, C] or [T, C, H, W]
    clips_len: int
        Length of the sampled clips in frames
    n_clips: int, default 16
        Number of clips to sample
    stride: int, default 8
        Distance in frames 
    Returns
    -------
    List[torch.Tensor]
        List of tensors containing `n_clips` of shape [clips_len, H, W, C]
    """
    n_video_frames = video.size(0)
    max_clips = n_video_frames // (clips_len + stride)

    if n_clips > max_clips:
        n_clips = max_clips

    start_idx = torch.randint(0, 
        high=(n_video_frames - (clips_len + stride) * n_clips) + 1,
        size=(1,)).item()
    
    clips_start_idx = torch.arange(
        start_idx, n_video_frames,  stride + clips_len)[:n_clips]
    
    return [video[i: i + clips_len] for i in clips_start_idx.tolist()]
