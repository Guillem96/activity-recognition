from typing import List, Tuple

import torch


def _align_video(video: torch.Tensor, src_fmt: str,
                 dst_fmt: str) -> torch.Tensor:
    """
    Examples
    --------
    >>> video = torch.randn(16, 112, 112, 3)
    >>> src_fmt = 'THWC'
    >>> dst_fmt = 'CTHW'
    >>> video = _align_video(video, src_fmt, dst_fmt)
    >>> video.size()
    torch.Size([3, 16, 112, 112])
    """

    assert len(src_fmt) == 4 and len(dst_fmt) == 4, "Formats must have 4 chars"
    assert len(set('THWC').intersection(src_fmt)) == 4, \
        "Format should contain TCHW"
    assert len(set('THWC').intersection(dst_fmt)), \
        "Format should contain TCHW"

    if src_fmt == dst_fmt:
        return video

    src_H_idx = src_fmt.index('H')
    src_W_idx = src_fmt.index('W')
    src_T_idx = src_fmt.index('T')
    src_C_idx = src_fmt.index('C')

    dst_H_idx = dst_fmt.index('H')
    dst_W_idx = dst_fmt.index('W')
    dst_T_idx = dst_fmt.index('T')
    dst_C_idx = dst_fmt.index('C')

    permutation = [0] * 4
    permutation[dst_H_idx] = src_H_idx
    permutation[dst_W_idx] = src_W_idx
    permutation[dst_T_idx] = src_T_idx
    permutation[dst_C_idx] = src_C_idx

    return video.permute(*permutation)


def uniform_sampling(video: torch.Tensor,
                     clips_len: int,
                     frames_stride: int = 1,
                     n_clips: int = 10,
                     overlap: bool = True,
                     video_fmt: str = 'THWC') -> List[torch.Tensor]:
    """
    Uniformly samples `n_clips` from a video. The clips samples by default can 
    be overlapping.

    Parameters
    ----------
    video: torch.Tensor
        Tensor of shape according to `video_fmt`
    clips_len: int
        Length of the sampled clips in frames
    frames_stride: int, default 1
        Separation between clips within a clip.
    n_clips: int, default 3
        Number of clips to sample
    overlap: bool, default True
        If set to False the sampled clips won't be overlapping
    video_fmt: str, default 'THWC'
        Format of the given video. Each character in the string indicates the 
        meaning of each dimension. For instance the format 'CTHW' specifies that
        the dimensions are the channels, timestamps, height and with 
        respectively. video_fmt has only 4 possible values {'T', 'C', 'H', 'W'}.

    Returns
    -------
    List[torch.Tensor]
        List of tensors containing `n_clips` of shape according to video_fmt, 
        where T will be `clips_len`
    """
    dst_fmt = 'THWC'
    video = _align_video(video, video_fmt, dst_fmt)

    n_frames = video.size(0)
    if overlap:
        indices = torch.randint(high=n_frames - (clips_len * frames_stride),
                                size=(n_clips, ))
        indices = indices.tolist()
    else:
        possible_indices = torch.arange(0, n_frames,
                                        clips_len * frames_stride)[:-1]
        choices = torch.randint(high=possible_indices.size(0),
                                size=(n_clips, ))
        indices = possible_indices[choices].tolist()

    clips = [
        video[i:i + (clips_len * frames_stride)][::frames_stride]
        for i in indices
    ]
    clips = [_align_video(o, dst_fmt, video_fmt) for o in clips]

    return clips


def lrcn_sampling(video: torch.Tensor,
                  clips_len: int,
                  n_clips: int = 16,
                  stride: int = 8,
                  video_fmt: str = 'THWC') -> List[torch.Tensor]:
    """
    Sample clips as described in LRCN paper. 
    Citation: At test time, we extract 16 frame clips with a stride of 8 frames
    from each video.

    If n_clips is larger than the total clips available in video the resulting
    list will have at most the clips available in the video.

    Parameters
    ----------
    video: torch.Tensor
        Tensor of shape according to `video_fmt`
    clips_len: int
        Length of the sampled clips in frames
    n_clips: int, default 16
        Number of clips to sample
    stride: int, default 8
        Distance in frames 
    video_fmt: str, default 'THWC'
        Format of the given video. Each character in the string indicates the 
        meaning of each dimension. For instance the format 'CTHW' specifies that
        the dimensions are the channels, timestamps, height and with 
        respectively. video_fmt has only 4 possible values {'T', 'C', 'H', 'W'}.

    Returns
    -------
    List[torch.Tensor]
        List of tensors containing `n_clips` of shape equal to the specified in
        `video_fmt`, where T will be equal to `clips_len`
    """
    dst_fmt = 'THWC'
    video = _align_video(video, video_fmt, dst_fmt)

    n_video_frames = video.size(0)
    max_clips = n_video_frames // (clips_len + stride)

    if n_clips > max_clips:
        n_clips = max_clips

    start_idx = torch.randint(0,
                              high=(n_video_frames -
                                    (clips_len + stride) * n_clips) + 1,
                              size=(1, )).item()

    clips_start_idx = torch.arange(start_idx, n_video_frames,
                                   stride + clips_len)[:n_clips]

    return [
        _align_video(video[i:i + clips_len], dst_fmt, video_fmt)
        for i in clips_start_idx.tolist()
    ]


def FstCN_sampling(video: torch.Tensor,
                   clips_len: int,
                   n_clips: int = 16,
                   n_crops: int = 9,
                   frames_stride: int = 1,
                   crops_size: Tuple[int, int] = (224, 224),
                   overlap: bool = False,
                   video_fmt: str = 'THWC') -> List[List[torch.Tensor]]:
    """
    Uniformly sampling `n_clips` from a given video, and for each clip, we
    randomly sample `n_crops` different crops.

    Parameters
    ----------
    video: torch.Tensor
        Tensor of shape according to `video_fmt`
    clips_len: int
        Length of the sampled clips in frames
    n_clips: int, default 16
        Number of clips to sample
    n_crops: int, default 0
        Number of crops to extract for each video clip
    frames_stride: int, default 1
        Distance between frames within a clip
    crops_size: Tuple[int, int], default (224, 224)
        Size of the generated crops
    overlap: bool, default True
        When sampling clips, can clips be overlapping
    video_fmt: str, default 'THWC'
        Format of the given video. Each character in the string indicates the 
        meaning of each dimension. For instance the format 'CTHW' specifies that
        the dimensions are the channels, timestamps, height and with 
        respectively. video_fmt has only 4 possible values {'T', 'C', 'H', 'W'}.

    Returns
    -------
    List[List[torch.Tensor]]
        Each list item contains a list of clips cropped at different points
    """
    import ar.transforms as VT
    import torchvision.transforms as T

    def to_video(o: torch.Tensor) -> torch.Tensor:
        return _align_video(o, 'CTHW', video_fmt).mul(255).byte()

    dst_fmt = 'THWC'
    video = _align_video(video, video_fmt, dst_fmt)
    W_idx = video_fmt.index('W')

    crop_fn = T.Compose(
        [VT.VideoToTensor(),
         VT.VideoRandomCrop(crops_size), to_video])

    clips = uniform_sampling(video=video,
                             clips_len=clips_len,
                             frames_stride=frames_stride,
                             n_clips=n_clips,
                             overlap=overlap,
                             video_fmt=dst_fmt)

    results = []
    for clip in clips:
        cropped_clips = [crop_fn(clip) for i in range(n_crops)]
        flipped_clips = [o.flip(dims=[W_idx]) for o in cropped_clips]
        results.append(cropped_clips + flipped_clips)

    return results
