import json
from datetime import datetime
from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
import functools

import click
import torch
import tqdm.auto as tqdm

import ar
from ar.typing import PathLike

_AVAILABLE_DATASETS = {'kinetics400', 'UCF-101'}
_AVAILABLE_MODELS = {'LRCN', 'FstCN', 'R2plus1', 'R2plus1_18', 'SlowFast'}

_ClipSamplerFn = Callable[[ar.io.VideoFramesIterator, int], List[torch.Tensor]]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _single_video_eval(path: ar.typing.PathLike,
                       label: int,
                       model: torch.nn.Module,
                       sampler: _ClipSamplerFn,
                       clips_len: int = 16,
                       transforms: Optional[ar.typing.Transform] = None,
                       batch_size: int = 16,
                       frame_rate: Optional[float] = None) -> Dict[str, float]:

    vit = ar.io.VideoFramesIterator(path,
                                    transforms=transforms,
                                    frame_rate=frame_rate)

    clips = sampler(vit, clips_len)
    clips = list(filter(lambda c: c.size(1) == clips_len, clips))
    if not clips:
        return None

    clips = torch.stack(clips)
    clips_ds = torch.utils.data.TensorDataset(clips)
    clips_dl = torch.utils.data.DataLoader(
        clips_ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=torch.utils.data.SequentialSampler(clips_ds))

    scores = []
    for clip, *_ in clips_dl:
        with torch.no_grad():
            probs = model(clip.to(device))
        scores.append(probs)

    scores_t = torch.cat(scores, dim=0)
    sci_fused_score = ar.video.fusion.SCI_fusion(scores_t, log_probs=True)
    avg_fused_score = torch.mean(scores_t, dim=0)

    sci_label = sci_fused_score.argmax(-1).item()
    avg_label = avg_fused_score.argmax(-1).item()

    return {
        'sci': {
            'is_correct': sci_label == label,
            'ground_truth': label,
            'predicted': sci_label,
        },
        'avg': {
            'is_correct': avg_label == label,
            'ground_truth': label,
            'predicted': avg_label,
        }
    }


@click.command()
@click.option('--dataset',
              type=click.Choice(list(_AVAILABLE_DATASETS)),
              required=True)
@click.option('--data-dir',
              type=click.Path(exists=True, file_okay=False),
              required=True,
              help='Dataset formated as imagenet directories')
@click.option('--annots-dir',
              type=click.Path(exists=True, file_okay=False),
              default=None,
              help='Dataset annotations. It is not needed for all'
              ' datasets, for now it is only required with '
              'UCF-101')
@click.option('--model',
              type=click.Choice(list(_AVAILABLE_MODELS)),
              required=True)
@click.option('--checkpoint', type=str, required=True)
@click.option('--output', type=click.Path(file_okay=False), default='reports')
def cli_video_level_eval(dataset: str, data_dir: str, annots_dir: str,
                         model: str, checkpoint: str, output: PathLike) -> None:
    ar.engine.seed()

    current_time = datetime.now().strftime('%b%d_%H_%M_%S')

    output = Path(output)
    output.mkdir(exist_ok=True, parents=True)
    output = output / f'results-{current_time}-{model}-{checkpoint.replace(".", "-")}.json'

    if dataset == 'UCF-101' and annots_dir is None:
        raise ValueError('"annots_dir" cannot be None when selecting '
                         'UCF-101 dataset')

    tfms = ar.transforms.valid_tfms()
    sampler_fn = functools.partial(ar.video.uniform_sampling, overlap=False)

    print('Creating dataset... ', end='')
    ds: Optional[ar.data.datasets.VideoLevelDataset] = None
    if dataset == 'UCF-101':
        ds = ar.data.VideoLevelUCF101(data_dir, annots_dir, 'test')
    elif dataset == 'kinetics400':
        ds = ar.data.VideoLevelKinetics(data_dir, 'test')
    else:
        raise ValueError('Unexpected dataset type')
    print('done')
    print('Loading model... ', end='')
    model_pt: Optional[ar.utils.checkpoint.SerializableModule] = None
    clips_len = 16
    if model == 'LRCN':
        clips_len = 16
        model_pt = ar.video.LRCN.from_pretrained(checkpoint,
                                                 map_location=device)
        model_pt.eval()
        model_pt.to(device)

    elif model == 'FstCN':
        clips_len = 16
        model_pt = ar.video.FstCN.from_pretrained(checkpoint,
                                                  map_location=device)
        model_pt.eval()
        model_pt.to(device)
        clips_len = (16 + model_pt.dt) * model_pt.st

    elif model == 'R2plus1' or model == 'R2plus1_18':
        clips_len = 16
        model_pt = ar.video.R2plus1_18.from_pretrained(checkpoint,
                                                       map_location=device)
        model_pt.eval()
        model_pt.to(device)

    elif model == 'SlowFast':
        desired_slow_frames = 4
        model_pt = ar.video.SlowFast.from_pretrained(checkpoint,
                                                     map_location=device)
        clips_len = desired_slow_frames * model_pt.tau
        model_pt.eval()
        model_pt.to(device)

    else:
        raise ValueError('Unexpected model architecture')
    print('done')

    print('=== Model HyperParameters ===')
    print(model_pt.config())

    sci_results = []
    avg_results = []

    for path, label in tqdm.tqdm(ds):
        pred = _single_video_eval(path=path,
                                  label=label,
                                  model=model_pt,
                                  sampler=sampler_fn,
                                  transforms=tfms,
                                  clips_len=clips_len,
                                  batch_size=16)
        if pred:
            sci_results.append(pred['sci'])
            avg_results.append(pred['avg'])

    total_len = len(ds)
    sci_correct = sum(o['is_correct'] for o in sci_results)
    avg_correct = sum(o['is_correct'] for o in avg_results)

    sci_acc = float(sci_correct) / total_len
    avg_acc = float(avg_correct) / total_len

    print(f'SCI Fused accuracy: {sci_acc}')
    print(f'Average Fused accuracy: {avg_acc}')

    report = {
        'config': model_pt.config(),
        'predictions': {'avg': avg_results, 'sci': sci_results},
        'checkpoint': checkpoint,
    }

    output.write_text(json.dumps(report))


if __name__ == "__main__":
    cli_video_level_eval()
