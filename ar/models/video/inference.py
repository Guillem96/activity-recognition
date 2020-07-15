import click
from typing import Optional

import torch
import torchvision.io as io
import torchvision.transforms as T

import ar
import ar.transforms as VT
from ar.utils.checkpoint import SerializableModule

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option('--video', '-v', help='Path to the video', required=True,
              type=click.Path(exists=True, dir_okay=False))
@click.option('--checkpoint', '-c', help='Path to the model checkpoint', 
              required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('--model', help='Model architecture', 
              required=True, type=click.Choice(['LRCN']))
@click.option('--class-names', help='Path to the class names file', 
              default=None, type=click.Path(exists=True, dir_okay=False))
@click.option('--clips-len', type=int, default=16, 
              help='Clips length in frames')
@click.option('--n-clips', '-n', type=int, default=3,
              help='Number of clips to sample and average '
                   'the resulting probability distributions')
def main(video: str, 
         checkpoint: str, 
         model: str, 
         class_names: str,
         clips_len: int, 
         n_clips: int) -> None:
    
    tfms = T.Compose([
        VT.VideoToTensor(),
        VT.VideoResize((224, 224)),
        VT.VideoNormalize(**VT.imagenet_stats)
    ])

    # Read the video
    video_t, *_ = io.read_video(video)

    # Sample non overlapping clips
    clips = ar.video.uniform_sampling(video=video_t, 
                                      clips_len=clips_len, 
                                      n_clips=n_clips)
    input_clips = torch.stack([tfms(o) for o in clips])

    # Load the model
    video_classifier: Optional[SerializableModule] = None

    if model == 'LRCN':
        video_classifier = ar.video.LRCN.from_pretrained(checkpoint, 
                                                         map_location=device)
    else:
        raise ValueError(f'Invalid model {model}')

    video_classifier.eval()

    # Make predictions
    with torch.no_grad():
        prob_dist = video_classifier(input_clips)
        prob_dist = prob_dist.exp().mean(0)
        score, label = prob_dist.max(0)

    # Load class mapping
    if class_names is not None:
        with open(class_names) as f:
            classes = list(f.readlines())
    else:
        classes = None
    
    if classes is not None:
        print(f'Model predicted {classes[label.item()]} '
              f'with {score.item()} of confidence')
    else:
        print(f'Model predicted {label.item()} '
              f'with {score.item()} of confidence')


if __name__ == "__main__":
    main()
