import torch
from torchvision.datasets.video_utils import VideoClips

import ar

dt = 9
st = 5
f = 16
# path = '/mnt/c/Users/Guillem/Pictures/Camera Roll/WIN_20210509_12_14_54_Pro.mp4'

# clips = VideoClips([path], clip_length_in_frames=(f + dt) * st)
# video, audio, info, idx = clips.get_clip(0)
# video = video[::st]
# print(video.size())
# video_idx = torch.arange(f)
# v = video[video_idx]
# v_diff = video[video_idx] - video[video_idx + dt]
# print(v.size(), v_diff.size())

clips = torch.randn(2, 3, (f + dt) * st, 112, 112)
model = ar.video.FstCN('resnet18', 10, pretrained=False)

with torch.no_grad():
    out = model(clips)
    print(out.size())



