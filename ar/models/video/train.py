import click
from typing import Any, Tuple, Optional, Union

import torch
import torch.optim as optim
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
import torchvision.transforms as T
from torchvision.datasets.samplers import UniformClipSampler, RandomClipSampler

from ar import engine
import ar.transforms as VT
from ar.metrics import accuracy
from ar.typing import Transform, SubOrDataset
from ar.utils.nn import _FEATURE_EXTRACTORS
from ar.utils.checkpoint import SerializableModule

from .models import LRCNN


_AVAILABLE_DATASETS = {'kinetics400', 'UCF-101'}
_AVAILABLE_MODELS = {'LRCNN',}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_datasets(
        dataset_type: str, 
        root: str, 
        frames_per_clip: int,
        workers: int = 1,
        annotations_path: str = None,
        validation_size: float = .1,
        train_transforms: Transform = None,
        valid_transforms: Transform = None) \
            -> Tuple[SubOrDataset, SubOrDataset]:
    """
    Given a dataset type, performs a set of operations to generate a train
    and validation dataset
    """
    def train_val_split(ds):
        rand_idx = torch.randperm(len(ds))
        train_len = int(len(ds) * validation_size)
        
        train_ds = Subset(ds, rand_idx[:train_len])
        train_ds.transform = train_transforms
        
        valid_ds = Subset(ds, rand_idx[train_len:])
        valid_ds.transform = train_transforms

        return train_ds, valid_ds

    if dataset_type == 'kinetics400':
        ds = torchvision.datasets.Kinetics400(
            root=root, frames_per_clip=frames_per_clip, extensions=('.mp4',),
            num_workers=workers)
        
        train_ds, valid_ds = train_val_split(ds)

    elif dataset_type == 'UCF-101':
        if annotations_path is None:
            raise ValueError(f'The annotations must be provided when using '
                             f'{dataset_type}')
            
        train_ds = torchvision.datasets.UCF101(
            root=root, annotation_path=annotations_path,
            step_between_clips=16, 
            frames_per_clip=frames_per_clip, train=True, 
            transform=train_transforms, num_workers=workers)

        valid_ds = torchvision.datasets.UCF101(
            root=root, annotation_path=annotations_path, 
            frames_per_clip=frames_per_clip, train=False,
            step_between_clips=16,
            transform=valid_transforms, num_workers=workers)

    return train_ds, valid_ds


def train(model: SerializableModule, 
          train_dl: DataLoader, 
          valid_dl: DataLoader,
          train_from: dict,
          **kwargs: Any) -> None:
    """
    Trains the models along specified epochs with the given train and validation
    dataloader.
    """
    criterion_fn = torch.nn.NLLLoss()
    optimizer, scheduler = load_optimizer(model, 
                                          train_from, 
                                          len(train_dl),
                                          **kwargs)
    starting_epoch = train_from.get('epoch', -1) + 1

    for epoch in range(starting_epoch, kwargs['epochs']):
        engine.train_one_epoch(dl=train_dl,
                               model=model,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               loss_fn=criterion_fn,
                               epoch=epoch,
                               print_freq=kwargs['print_freq'],
                               device=device)
        
        engine.evaluate(dl=valid_dl,
                        model=model,
                        metrics=[accuracy],
                        loss_fn=criterion_fn,
                        device=device)
        
        # Save the model jointly with the optimizer
        model.save(
            kwargs['save_checkpoint'],
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else {})


def data_preparation(**kwargs: Any) -> Tuple[DataLoader, DataLoader]:
    """
    Loads the datasets with corresponding transformations and creates two data
    loaders, one for train and validation
    """
    def collate_fn(batch):
        batch = [(d[0], d[2]) for d in batch]
        return default_collate(batch)
        
    train_tfms = T.Compose([
        VT.VideoToTensor(),
        VT.VideoResize((224, 224)),
        VT.VideoRandomHorizontalFlip(),
        VT.VideoNormalize(**VT.imagenet_stats),
    ])

    valid_tfms = T.Compose([
        VT.VideoToTensor(),
        VT.VideoResize((224, 224)),
        VT.VideoNormalize(**VT.imagenet_stats),
    ])

    train_ds, valid_ds = load_datasets(kwargs['dataset'], kwargs['data_dir'],
                                       kwargs['frames_per_clip'], 
                                       kwargs['data_loader_workers'],
                                       kwargs['annots_dir'], 
                                       kwargs['validation_split'],
                                       train_tfms, valid_tfms)

    if (isinstance(train_ds, Subset) and 
            hasattr(train_ds.dataset, 'video_clips')):
        train_sampler = RandomClipSampler(train_ds.dataset.video_clips, 10)
        valid_sampler = UniformClipSampler(valid_ds.dataset.video_clips, 10)
    elif hasattr(train_ds, 'video_clips'):
        train_sampler = RandomClipSampler(train_ds.video_clips, 10)
        valid_sampler = UniformClipSampler(valid_ds.video_clips, 10)
    else:
        raise ValueError('Video dataset must have the video_clips attribute')

    train_dl = DataLoader(train_ds, 
                          batch_size=kwargs['batch_size'],
                          num_workers=kwargs['data_loader_workers'],
                          sampler=train_sampler, 
                          collate_fn=collate_fn,
                          pin_memory=True)

    valid_dl = DataLoader(valid_ds, 
                          batch_size=kwargs['batch_size'],
                          num_workers=kwargs['data_loader_workers'],
                          sampler=valid_sampler,
                          collate_fn=collate_fn,
                          pin_memory=True)

    return train_dl, valid_dl


def load_optimizer(
        model: torch.nn.Module, 
        checkpoint: dict,
        steps_per_epoch: int = -1,
        **kwargs: Any) -> Tuple[optim.Optimizer, 
                                Optional[optim.lr_scheduler._LRScheduler]]:
    """
    Load an optimizer to update the model parameters and if specified,
    also creates a learning rate scheduler.

    If a checkpoint dict is provided, this loads the optimizer and scheduler 
    state dicts
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    if kwargs['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(trainable_params, 
                                lr=kwargs['learning_rate'])
    elif kwargs['optimizer'] == 'SGD':
        optimizer = optim.SGD(trainable_params, lr=kwargs['learning_rate'],
                              momentum=0.9, weight_decay=4e-5)
    elif kwargs['optimizer'] == 'Adam':
        optimizer = optim.Adam(trainable_params, lr=kwargs['learning_rate'])
    
    if kwargs['scheduler'] == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, kwargs['learning_rate'] * 10, 
            steps_per_epoch * kwargs['epochs'])
    elif kwargs['scheduler'] == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, steps_per_epoch, .1)
    elif kwargs['scheduler'] == 'None':
        scheduler = None
    
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if 'scheduler' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return optimizer, scheduler


@click.group()

# Data options
@click.option('--dataset', type=click.Choice(list(_AVAILABLE_DATASETS)), 
              required=True)
@click.option('--data-dir', type=click.Path(exists=True, file_okay=False),
              required=True, help='Dataset formated as imagenet directories')
@click.option('--annots-dir', type=click.Path(exists=True, file_okay=False),
              default=None, help='Dataset annotations. It is not needed for all'
                                 ' datasets, for now it is only required with '
                                 'UCF-101')
@click.option('--validation-split', type=float, default=.1)
@click.option('--data-loader-workers', type=int, default=2)
@click.option('--frames-per-clip', type=int, required=True)

# Training behavior
@click.option('--epochs', type=int, default=10)
@click.option('--batch-size', type=int, default=16)

@click.option('--optimizer', type=click.Choice(['SGD', 'Adam', 'AdamW']),
              default='SGD')
@click.option('--grad-accum-steps', type=int, default=1)
@click.option('--learning-rate', type=float, default=1e-3)
@click.option('--scheduler', type=click.Choice(['OneCycle', 'Step', 'None']),
              default='None')

# Logging options
@click.option('--print-freq', type=int, default=20, 
              help='Print training epoch progress every n steps')

# Checkpointing options
@click.option('--resume-checkpoint', type=click.Path(dir_okay=False),
              default=None, help='Resume training from')
@click.option('--save-checkpoint', type=click.Path(dir_okay=False),
              default='models/model.pt', help='File to save the checkpoint')
@click.pass_context
def main(ctx: click.Context, **kwargs: Any) -> None:
    engine.seed()
    ctx.ensure_object(dict)

    train_dl, valid_dl = data_preparation(**kwargs)

    ctx.obj['common'] = kwargs
    ctx.obj['train_dl'] = train_dl
    ctx.obj['valid_dl'] = valid_dl


@main.command(name='LRCNN')

@click.option('--feature-extractor', 
              type=click.Choice(list(_FEATURE_EXTRACTORS)),
              default='resnet18')
@click.option('--freeze-fe/--no-freeze-fe', default=False,
              help='Wether or not to fine tune the pretrained'
                   ' feature extractor')
@click.option('--rnn-units', 
              type=int, default=512, 
              help='Hidden size of the LSTM layer added on top of the feature '
                   'extractors')
@click.option('--bidirectional/--no-bidirectional', 
              default=True, help='Wether to use a bidirectional LSTM or an '
                                 ' autoregressive')
@click.pass_context
def train_LRCNN(ctx: click.Context, **kwargs: Any) -> None:
    kwargs.update(ctx.obj['common'])

    train_dl, valid_dl = ctx.obj['train_dl'], ctx.obj['valid_dl']

    ds = train_dl.dataset
    if isinstance(ds, Subset):
        ds = ds.dataset

    if kwargs['resume_checkpoint'] is None:
        checkpoint: dict = dict()
        model = LRCNN(kwargs['feature_extractor'], 
                      len(ds.classes),
                      freeze_feature_extractor=kwargs['freeze_fe'],
                      rnn_units=kwargs['rnn_units'], 
                      bidirectional=kwargs['bidirectional'])
    else:
        model, checkpoint = LRCNN.load(
            kwargs['resume_checkpoint'],
            map_location=device,
            freeze_feature_extractor=kwargs['freeze_fe'])
        
    model.to(device)

    train(model, train_dl, valid_dl, train_from=checkpoint, **kwargs)


if __name__ == "__main__":
    main()
