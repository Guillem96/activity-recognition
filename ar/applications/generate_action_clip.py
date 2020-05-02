from typing import Tuple
from pathlib import Path

import click

import torch
import torchvision

import ar.transforms as T 
from ar.models.image.classifier import ImageClassifier


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@torch.no_grad()
def process_video(video: torch.FloatTensor,
                  image_classifier: torch.nn.Module) -> torch.FloatTensor:
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
    torch.FloatTensor
        Model outputs for each frame
    """
    batch_size = 64
    predictions = []
    for i in range(0, video.shape[0], batch_size):
        inp = video[i: i + batch_size]
        preds = image_classifier(inp.to(device))
        predictions.append(preds.cpu())
    
    return torch.cat(predictions, dim=0)


def find_video_clips(
        video: torch.FloatTensor, 
        image_classifier: torch.nn.Module,
        topk: int = 2) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """
    Find frames with topk probabilities to contain a class.

    Returns
    -------
    Tuple[torch.LongTensor, torch.LongTensor]
        Return the frame indices with largest probabilites and the labels of the 
        respective frames
    """
    outputs = process_video(video, image_classifier)
    max_elems = outputs.max(dim=-1)
    max_probs = max_elems.values
    labels = max_elems.indices
    topk_indices = max_probs.topk(topk).indices
    return topk_indices, labels[topk_indices]


@click.command()

@click.option('--video-path', type=click.Path(dir_okay=False, exists=True),
              required=True, help='Video to extract the clips from')
@click.option('--skip-frames', type=int, default=2)
@click.option('--image-classifier-checkpoint', 
              type=click.Path(dir_okay=False, exists=True),
              required=True, help='Path to the pretrained model')
@click.option('--class-names', 
              type=click.Path(dir_okay=False, exists=True),
              required=True, help='Path to the class names file')
@click.option('--out-dir', required=True, 
              type=click.Path(file_okay=False))
@click.option('--n-clips', type=int, default=2, 
              help='Number of clips generated')
@click.option('--clip-len', type=int, default=5,
              help='Length of the extracted clips')
def main(video_path, 
         skip_frames, 
         image_classifier_checkpoint, 
         class_names,
         out_dir,
         n_clips,
         clip_len):

    def frame_to_second(frame_idx: int, video_info: dict) -> float:
        real_frame = frame_idx * skip_frames
        return real_frame / info['video_fps']

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    classes = Path(class_names).read_text().split('\n')

    print(f'Loading image classifier from {image_classifier_checkpoint}...')
    model, _ = ImageClassifier.load(image_classifier_checkpoint,
                                    map_location=device)
    model.eval()
    model.to(device)

    print(f'Loading video {video_path}...')
    video, _, info = torchvision.io.read_video(video_path)
    video_duration = video.size(0) / info['video_fps']
    frames_seconds_after_skip = info['video_fps'] / 4
    video_t = video[::skip_frames]

    # Clip duration extra variables
    clip_duration_frames = info['video_fps'] * clip_len
    
    # Resize and normalize the video
    tfms = torchvision.transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((128, 128)),
        T.Normalize(**T.imagenet_stats)
    ])

    video_t = tfms(video_t)
    video_t = video_t.permute(1, 0, 2, 3)
    
    print(f'Processing video...')
    res = find_video_clips(video_t, model, topk=n_clips)
    
    for frame_idx, label in zip(*res):
        label = classes[label.item()]

        # Getting video seconds of the clip just for logging purposes
        video_second = frame_to_second(frame_idx, info)
        clip_start = max(0, video_second - clip_len / 2.)
        clip_end = min(video_duration, video_second + clip_len / 2.)
        print(f'Clip with label {label} starts at {clip_start:.2f} '
              f'and ends at {clip_end:.2f}')
        
        # Get clip frames ranges and save the extracted clip
        original_frame = frame_idx * skip_frames
        start_frame = int(max(0, frame_idx - clip_duration_frames / 2))
        end_frame = int(min(video.size(0), frame_idx + clip_duration_frames / 2))
        clip_fname = out_dir / f'{start_frame}_{end_frame}_{label}.mp4'
        
        print(f'Saving video at {str(clip_fname)}')
        torchvision.io.write_video(str(clip_fname), 
                                   video[start_frame: end_frame],
                                   info['video_fps'])


if __name__ == "__main__":
    main()