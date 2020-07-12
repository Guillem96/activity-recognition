import click
from typing import Optional
import tqdm.auto as tqdm

import torch
import torchvision.transforms as T

import ar
import ar.transforms as VT

_AVAILABLE_DATASETS = {'kinetics400', 'UCF-101'}
_AVAILABLE_MODELS = {'LRCN',}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@click.command()
@click.option('--dataset', type=click.Choice(list(_AVAILABLE_DATASETS)),
              required=True)
@click.option('--data-dir', type=click.Path(exists=True, file_okay=False),
              required=True, help='Dataset formated as imagenet directories')
@click.option('--annots-dir', type=click.Path(exists=True, file_okay=False),
              default=None, help='Dataset annotations. It is not needed for all'
                                 ' datasets, for now it is only required with '
                                 'UCF-101')

@click.option('--model-arch', type=click.Choice(list(_AVAILABLE_MODELS)),
              required=True)
@click.option('--checkpoint', type=str, required=True)
def video_level_eval(dataset: str, data_dir: str, annots_dir: str,
                     model_arch: str, checkpoint: str) -> None:
    
    if dataset == 'UCF-101' and annots_dir is None:
        raise ValueError('"annots_dir" cannot be None when selecting '
                         'UCF-101 dataset')
    
    tfms = T.Compose([
        VT.VideoToTensor(),
        VT.VideoResize((128, 171)),
        VT.VideoCenterCrop((112, 112)),
        VT.VideoNormalize(**VT.imagenet_stats)
    ])

    ds: Optional[ar.data.datasets.VideoLevelDataset] = None
    if dataset == 'UCF-101':
        ds = ar.data.VideoLevelUCF101(data_dir, annots_dir, 'test', 
                                      transform=tfms)
    elif dataset == 'kinetics400':
        ds = ar.data.VideoLevelKinetics(data_dir, 'test', transform=tfms)
    
    # model: Optional[ar.checkpoint.SerializableModule] = None
    if model_arch == 'LRCN':
        model = ar.video.LRCN.from_pretrained(checkpoint)
        model.to(device)
    
    clips_sampler = ar.video.lrcn_sampling
    metrics = [ar.metrics.accuracy, 
               ar.metrics.top_3_accuracy, 
               ar.metrics.top_5_accuracy]

    final_preds = []
    for i in tqdm.trange(len(ds)):
        video, audio, label = ds[i]
        label = torch.as_tensor(label).to(device).view(-1, 1)
        clips = torch.stack(clips_sampler(video.permute(1, 0, 2, 3), 16))
        clips = clips.permute(0, 2, 1, 3, 4).to(device)

        with torch.no_grad():
            predictions = model(clips)
            predictions = predictions.mean(0, keepdim=True)
            final_preds.append(predictions.cpu())
    
    final_preds_t = torch.stack(final_preds)
    final_metrics = {m.__name__: m(final_preds_t, torch.as_tensor(ds.labels)) 
                     for m in metrics}
    final_metrics = ', '.join(f'{k}: {v:.4f}' for k, v in final_metrics.items())
    print(final_metrics) 


if __name__ == "__main__":
    video_level_eval()