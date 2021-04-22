import gc
from pathlib import Path
from typing import List, Tuple, Union

import click
import tqdm.auto as tqdm

import torch
import torchvision

import ar.transforms as T
from ar.typing import Transform
from ar.models.image.classifier import ImageClassifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TODO: Implement with videoreader api torchvision.io


@torch.no_grad()
def process_video(video: torch.Tensor,
                  image_classifier: torch.nn.Module) -> torch.Tensor:
    """
    Maps the image classifier for each video frame and return the classifier
    outputs

    Parameters
    ----------
    video: torch.Tensor of shape [FRAMES, 3, HEIGHT, WIDTH]
        Video to process
    image_classifier: torch.nn.Module
        Model trained for image classification
    
    Returns
    -------
    torch.Tensor
        Model outputs for each frame
    """
    batch_size = 64
    predictions = []
    for i in range(0, video.shape[0], batch_size):
        inp = video[i:i + batch_size]
        preds = image_classifier(inp.to(device))
        predictions.append(preds.cpu())

    return torch.cat(predictions, dim=0)


def find_video_clips(video: torch.Tensor,
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
    outputs = process_video(video, image_classifier)
    max_probs, labels = outputs.max(dim=-1)
    _, topk_indices = max_probs.topk(topk)
    return topk_indices, labels[topk_indices]


def _process_videofile(video_path: Path, skip_frames: int, clip_len: int,
                       n_clips: int, tfms: Transform, model: torch.nn.Module,
                       classes: List[str], out_dir: Path):

    video, _, info = torchvision.io.read_video(str(video_path))
    video_t = video[::skip_frames]

    # Clip duration extra variables
    clip_duration_frames = info['video_fps'] * clip_len

    video_t = tfms(video_t)
    video_t = video_t.permute(1, 0, 2, 3)

    res = find_video_clips(video_t, model, topk=n_clips)

    for frame_idx, label in zip(*res):
        label_name = classes[int(label.item())].replace(' ', '_')
        interpolated_fi = frame_idx * skip_frames

        # Get clip frames ranges and save the extracted clip
        start_frame = int(max(0, interpolated_fi - clip_duration_frames / 2))
        end_frame = int(
            min(video.size(0), interpolated_fi + clip_duration_frames / 2))

        clip_out_dir = out_dir / label_name
        clip_out_dir.mkdir(exist_ok=True)
        clip_fname = clip_out_dir / f'{start_frame}_{end_frame}.mp4'

        torchvision.io.write_video(str(clip_fname),
                                   video[start_frame:end_frame],
                                   int(info['video_fps']))


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
def main(video_path: Union[str, Path], skip_frames: int,
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
                               clip_len=clip_len,
                               n_clips=n_clips,
                               tfms=tfms,
                               model=model,
                               out_dir=out_dir,
                               classes=classes)
            gc.collect()


if __name__ == "__main__":
    main()
