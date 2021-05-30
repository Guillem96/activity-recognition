import json
from collections import OrderedDict
from pathlib import Path
from typing import List
from typing import Mapping
from typing import OrderedDict
from typing import Union

import click
import torch
import torch.nn.functional as F
import tqdm.auto as tqdm

import ar
import ar.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _compute_edges(current_key: str,
                   features: Mapping[str, torch.Tensor],
                   threshold: float = .97) -> List[str]:
    keys = list(features)
    source = features[current_key]
    others = torch.as_tensor([features[k] for k in features])

    similarities = F.cosine_similarity(source.unsqueeze(0), others)
    connects_with = (similarities > threshold).nonzero().flatten()

    return [keys[i] for i in connects_with if keys[i] != current_key]


@torch.no_grad()
def _compute_feature_vector(video_it: ar.io.VideoFramesIterator,
                            model: torch.nn.Module) -> torch.Tensor:
    _, first_batch = next(video_it)
    initial_features = model(first_batch).sum(0)
    n = first_batch.size(0)

    for _, batch in video_it:
        features = model(batch).sum(0)
        initial_features += features
        n += batch.size(0)

    return initial_features / n


@click.command()
@click.option('--video-path',
              type=click.Path(exists=True, file_okay=False),
              required=True,
              help='Video to extract the clips from')
@click.option('--feature-extractor',
              type=click.Choice(list(ar.nn._FEATURE_EXTRACTORS)),
              default='resnet18',
              help='Pretrained image feature extractor.')
@click.option('--skip-frames', type=int, default=2)
@click.option(
    '--batch-size',
    type=int,
    default=64,
    help=
    'Group frames in samples of n examples to speed up the image classifier inference time'
)
@click.option('--out', help='JSON file to serialize the graph', required=True)
def main(video_path: Union[str, Path], feature_extractor: str, skip_frames: int,
         batch_size: int, out: Union[str, Path]) -> None:

    out = Path(out)
    out.parent.mkdir(exist_ok=True, parents=True)

    video_path = Path(video_path)

    model, _ = ar.nn.image_feature_extractor(feature_extractor)
    model.eval()
    model.to(device)

    # Resize and normalize the video
    tfms = torch.nn.Sequential(T.VideoToTensor(), T.VideoResize((128, 128)),
                               T.VideoNormalize(**T.imagenet_stats))

    nodes = OrderedDict()
    videos = list(video_path.rglob("*.mp4"))
    for v_fname in tqdm.tqdm(videos, desc='Generating feature vectors'):
        video_it = ar.io.VideoFramesIterator(v_fname,
                                             transforms=tfms,
                                             batch_size=batch_size,
                                             skip_frames=skip_frames)
        key = f'{v_fname.parent}/{v_fname.name}'
        features = _compute_feature_vector(video_it, model)
        nodes[key] = features

    graph = OrderedDict({node: _compute_edges(node, nodes) for node in nodes})
    with out.open('w') as graph_f:
        json.dump(graph, graph_f)


if __name__ == '__main__':
    main()
