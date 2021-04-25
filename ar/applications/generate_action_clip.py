import gc
import itertools
from pathlib import Path
from typing import List, Optional, Tuple, Union

import click
import tqdm.auto as tqdm

import torch
import torchvision

import ar.transforms as T
from ar.typing import Transform
from ar.models.image import ImageClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class _VideoFramesIterator(object):
    def __init__(self,
                 video_path: Path,
                 batch_size: int,
                 skip_frames: int = 1,
                 transforms: Optional[Transform] = None) -> None:
        self.video_reader = torchvision.io.VideoReader(str(video_path))
        self.current_frame = 0
        self.skip_frames = skip_frames
        self.batch_size = batch_size
        self.tranforms = transforms or (lambda x: x)
        self._is_it_end = False
        self.metadata = self.video_reader.get_metadata()

    @property
    def video_fps(self) -> float:
        return self.metadata['video']['fps'][0]

    @property
    def video_duration(self) -> float:
        return self.metadata['video']['duration'][0]

    def take(self, from_sec: int, to_sec: int) -> torch.Tensor:
        video_it = self.video_reader.seek(from_sec)
        frames = [
            f['data']
            for f in itertools.takewhile(lambda x: x['pts'] < to_sec, video_it)
        ]
        return torch.stack(frames).permute(0, 2, 3, 1)

    def __iter__(self):
        return self

    def __next__(self) -> torch.Tensor:
        if self._is_it_end:
            raise StopIteration

        frames = []
        frames_idx = []
        start = self.current_frame
        end = start + self.batch_size * self.skip_frames

        for _ in range(start, end):
            try:
                frame = next(self.video_reader)
            except StopIteration:
                self._is_it_end = True
                break

            if self.current_frame % self.skip_frames == 0:
                frames.append(frame['data'])
                frames_idx.append(self.current_frame)

            self.current_frame += 1

        # (FRAMES, CHANNELS, HEIGHT, WIDTH) to (FRAMES, HEIGHT, WIDTH, CHANNELS)
        video_clip = torch.stack(frames).permute(0, 2, 3, 1)
        video_clip = self.tranforms(video_clip)
        frames_idx = torch.as_tensor(frames_idx, dtype=torch.long)

        return frames_idx, video_clip


@torch.no_grad()
def _process_video(video: _VideoFramesIterator,
                   image_classifier: torch.nn.Module) -> torch.Tensor:
    """
    Maps the image classifier for each video frame and return the classifier
    outputs

    Parameters
    ----------
    video: _VideoFramesIterator
        Video iterator
    image_classifier: torch.nn.Module
        Model trained for image classification
    
    Returns
    -------
    torch.Tensor
        Model outputs for each frame
    """
    predictions = []
    for _, video_clip in video:
        preds = image_classifier(video_clip.to(device))
        predictions.append(preds.cpu())

    return torch.cat(predictions, dim=0)


def _find_video_clips(video: torch.Tensor,
                      image_classifier: torch.nn.Module,
                      topk: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find frames with topk probabilities to contain a class.

    Returns
    -------
    Tuple[torch.LongTensor, torch.LongTensor]
        Return the frame indices with largest probabilites and the labels of the 
        respective frames
    """
    outputs = _process_video(video, image_classifier)
    max_probs, labels = outputs.max(dim=-1)
    _, topk_indices = max_probs.topk(topk)
    return topk_indices, labels[topk_indices]


def _process_videofile(video_path: Path, skip_frames: int, clip_len: int,
                       batch_size: int, n_clips: int, tfms: Transform,
                       model: torch.nn.Module, classes: List[str],
                       out_dir: Path):

    video_it = _VideoFramesIterator(video_path,
                                    batch_size=batch_size,
                                    transforms=tfms,
                                    skip_frames=skip_frames)

    res = _find_video_clips(video_it, model, topk=n_clips)

    for frame_idx, label in zip(*res):
        label_name = classes[int(label.item())].replace(' ', '_')
        interpolated_fi = frame_idx * skip_frames
        mid_point_seconds = interpolated_fi / video_it.video_fps

        # Get clip frames ranges and save the extracted clip
        start_seconds = int(max(0, mid_point_seconds - clip_len / 2))
        end_sec = int(
            min(video_it.video_duration, mid_point_seconds + clip_len / 2))

        clip_out_dir = out_dir / label_name
        clip_out_dir.mkdir(exist_ok=True)
        clip_fname = clip_out_dir / f'{start_seconds}_{end_sec}.mp4'

        clip_frames = video_it.take(start_seconds, end_sec)
        torchvision.io.write_video(str(clip_fname), clip_frames,
                                   int(video_it.video_fps))


@click.command()
@click.option('--video-path',
              type=click.Path(exists=True),
              required=True,
              help='Video to extract the clips from')
@click.option('--skip-frames', type=int, default=2)
@click.option('--image-classifier-checkpoint',
              type=click.Path(dir_okay=False, exists=True),
              required=True,
              help='Path to the pretrained model')
@click.option(
    '--batch-size',
    type=int,
    default=64,
    help=
    'Group frames in samples of n examples to speed up the image classifier inference time'
)
@click.option('--class-names',
              type=click.Path(dir_okay=False, exists=True),
              required=True,
              help='Path to the class names file')
@click.option('--out-dir', required=True, type=click.Path(file_okay=False))
@click.option('--n-clips',
              type=int,
              default=2,
              help='Number of clips generated')
@click.option('--clip-len',
              type=int,
              default=5,
              help='Length of the extracted clips in seconds')
def main(video_path: Union[str, Path], skip_frames: int, batch_size: int,
         image_classifier_checkpoint: str, class_names: Union[str, Path],
         out_dir: Union[str, Path], n_clips: int, clip_len: int) -> None:

    video_path = Path(video_path)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    classes = Path(class_names).read_text().split('\n')

    print(f'Loading image classifier from {image_classifier_checkpoint}...')
    model, _ = ImageClassifier.load(image_classifier_checkpoint,
                                    map_location=device)
    model.eval()
    model.to(device)

    # Resize and normalize the video
    tfms = torchvision.transforms.Compose([
        T.VideoToTensor(),
        T.VideoResize((128, 128)),
        T.VideoNormalize(**T.imagenet_stats)
    ])

    if video_path.is_file():
        _process_videofile(video_path,
                           skip_frames=skip_frames,
                           batch_size=batch_size,
                           clip_len=clip_len,
                           n_clips=n_clips,
                           tfms=tfms,
                           model=model,
                           out_dir=out_dir,
                           classes=classes)
    else:
        videos = list(video_path.rglob("*.mp4"))
        for v_fname in tqdm.tqdm(videos, desc='Generating clips'):
            _process_videofile(v_fname,
                               skip_frames=skip_frames,
                               batch_size=batch_size,
                               clip_len=clip_len,
                               n_clips=n_clips,
                               tfms=tfms,
                               model=model,
                               out_dir=out_dir,
                               classes=classes)
            gc.collect()


if __name__ == "__main__":
    main()
