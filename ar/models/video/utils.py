from typing import List

import torch


def uniform_sampling(video: torch.Tensor, 
                     clips_len: int, 
                     n_clips: int = 3,
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