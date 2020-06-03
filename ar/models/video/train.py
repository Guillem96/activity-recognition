import click
from typing import Any, Tuple, Optional, Union, Sequence, Type, Dict

import torch
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as T
from torchvision.datasets.samplers import UniformClipSampler, RandomClipSampler

import ar
import ar.transforms as VT
from ar.typing import Transform, Optimizer, Scheduler


_AVAILABLE_DATASETS = {'kinetics400', 'UCF-101'}
_AVAILABLE_MODELS = {'LRCN',}

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
            -> Tuple[ar.data.VideoDataset, ar.data.VideoDataset]:
    """
    Given a dataset type, performs a set of operations to generate a train
    and validation dataset
    """
    
    train_ds: Optional[ar.data.VideoDataset] = None
    valid_ds: Optional[ar.data.VideoDataset] = None
    if dataset_type == 'kinetics400':
        train_ds = ar.data.Kinetics400(
            root=root, split='train', 
            frames_per_clip=frames_per_clip, 
            extensions=('.mp4',),
            num_workers=workers,
            transform=train_transforms)
        
        valid_ds = ar.data.Kinetics400(
            root=root, split='validate', 
            frames_per_clip=frames_per_clip, 
            extensions=('.mp4',),
            num_workers=workers,
            transform=valid_transforms)

    else: # dataset_type == 'UCF-101':
        if annotations_path is None:
            raise ValueError(f'The annotations must be provided when using '
                             f'{dataset_type}')
            
        train_ds = ar.data.UCF101(
            root=root, annotation_path=annotations_path,
            frames_per_clip=frames_per_clip, split='train', 
            transform=train_transforms, num_workers=workers)

        valid_ds = ar.data.UCF101(
            root=root, annotation_path=annotations_path, 
            frames_per_clip=frames_per_clip, split='test',
            transform=valid_transforms, num_workers=workers)

    return train_ds, valid_ds


def data_preparation(**kwargs: Any) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Loads the datasets with corresponding transformations and creates two data
    loaders, one for train and validation
    """
    Collated = Union[Tuple[torch.Tensor, torch.Tensor], 
                     Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]]

    def collate_fn(batch: Sequence[Any]) -> Collated:
        video, audio, label, info = zip(*batch)
        
        if not kwargs['audio']:
            return torch.stack(video), torch.as_tensor(label)
        else:
            return ((torch.stack(video), torch.stack(audio)), 
                    torch.as_tensor(label))
        
    train_tfms = T.Compose([
        VT.VideoToTensor(),
        VT.OneOf([
            T.Compose([
                VT.VideoResize((300, 300)),
                VT.VideoRandomCrop((224, 224)),
            ]),
            VT.VideoCenterCrop((224, 224)),
            VT.VideoResize((224, 224))
        ]),
        VT.VideoRandomHorizontalFlip(),
        VT.VideoRandomErase(scale=(0.02, 0.15)),
        VT.VideoNormalize(**VT.imagenet_stats)
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

    if hasattr(train_ds, 'video_clips'):
        train_sampler = RandomClipSampler(train_ds.video_clips, 10)
        valid_sampler = UniformClipSampler(valid_ds.video_clips, 10)
    else:
        raise ValueError('Video dataset must have the video_clips attribute')

    train_dl = data.DataLoader(train_ds, 
                               batch_size=kwargs['batch_size'],
                               num_workers=kwargs['data_loader_workers'],
                               sampler=train_sampler,
                               collate_fn=collate_fn,
                               pin_memory=True)

    valid_dl = data. DataLoader(valid_ds, 
                                batch_size=kwargs['batch_size'],
                                num_workers=kwargs['data_loader_workers'],
                                sampler=valid_sampler,
                                collate_fn=collate_fn,
                                pin_memory=True)
 
    return train_dl, valid_dl


def train(model: ar.checkpoint.SerializableModule, 
          train_dl: data.DataLoader, 
          valid_dl: data.DataLoader,
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
    # TODO: Enable mixed precision when pytorch 1.6.0
    
    for epoch in range(starting_epoch, kwargs['epochs']):
        ar.engine.train_one_epoch(dl=train_dl,
                                  model=model,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  loss_fn=criterion_fn,
                                  epoch=epoch,
                                  grad_accum_steps=kwargs['grad_accum_steps'],
                                #   mixed_precision=kwargs['fp16'],
                                  print_freq=kwargs['print_freq'],
                                  device=device)
        
        ar.engine.evaluate(dl=valid_dl,
                           model=model,
                           metrics=[ar.metrics.accuracy, 
                                    ar.metrics.top_3_accuracy],
                           loss_fn=criterion_fn,
                        #    mixed_precision=kwargs['fp16'],
                           device=device)
        
        # Save the model jointly with the optimizer
        model.save(
            kwargs['save_checkpoint'],
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else {})


def load_optimizer(
        model: torch.nn.Module, 
        checkpoint: dict,
        steps_per_epoch: int = -1,
        **kwargs: Any) -> Tuple[Optimizer, 
                                Optional[Scheduler]]:
    """
    Load an optimizer to update the model parameters and if specified,
    also creates a learning rate scheduler.

    If a checkpoint dict is provided, this loads the optimizer and scheduler 
    state dicts
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer: Optimizer = None # type: ignore
    if kwargs['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(trainable_params, 
                                lr=kwargs['learning_rate'])
    elif kwargs['optimizer'] == 'SGD':
        optimizer = optim.SGD(trainable_params, lr=kwargs['learning_rate'],
                              momentum=0.9, weight_decay=4e-5)
    elif kwargs['optimizer'] == 'Adam':
        optimizer = optim.Adam(trainable_params, lr=kwargs['learning_rate'])
    
    if kwargs['scheduler'] == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR( # type: ignore
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


def _load_model(model_name: str,
                out_units: int,
                feature_extractor: str,
                freeze_fe: bool,
                resume_checkpoint: str = None,
                **kwargs: Any) \
                    -> Tuple[ar.checkpoint.SerializableModule, dict]:
    
    model_classes: Dict[str, Type[ar.checkpoint.SerializableModule]] = dict(
        LRCN=ar.video.LRCN,
        LRCNWithAudio=ar.video.LRCNWithAudio)

    model_cls = model_classes[model_name]

    if resume_checkpoint is None:
        checkpoint: dict = dict()
        model = model_cls(feature_extractor, 
                          out_units,
                          freeze_feature_extractor=freeze_fe,
                          **kwargs)
    else:
        model, checkpoint = model_cls.load(
            resume_checkpoint,
            map_location=device,
            freeze_feature_extractor=freeze_fe)

    return model, checkpoint


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

@click.option('--audio/--no-audio', 
              default=False, help='Use audio for training too')

# Training behavior
@click.option('--epochs', type=int, default=10)
@click.option('--batch-size', type=int, default=16)

@click.option('--optimizer', type=click.Choice(['SGD', 'Adam', 'AdamW']),
              default='SGD')
@click.option('--grad-accum-steps', type=int, default=1)
@click.option('--learning-rate', type=float, default=1e-3)
@click.option('--scheduler', type=click.Choice(['OneCycle', 'Step', 'None']),
              default='None')

# Training optimizations
@click.option('--fp16/--no-fp16', default=False,
              help='Perform the forward pass of the model and the loss '
                   'computation with mixed precision. The backward pass stays'
                   'with fp32')
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
    ar.engine.seed()
    ctx.ensure_object(dict)

    train_dl, valid_dl = data_preparation(**kwargs)

    ctx.obj['common'] = kwargs
    ctx.obj['train_dl'] = train_dl
    ctx.obj['valid_dl'] = valid_dl


@main.command(name='LRCN')

@click.option('--feature-extractor', 
              type=click.Choice(list(ar.nn._FEATURE_EXTRACTORS)),
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

@click.option('--audio-features', 
              type=int, default=512, 
              help='Number of features to compute for audio')
@click.option('--mel-features', 
              type=int, default=40, 
              help='N mels to extract on MFCC')

@click.pass_context
def train_LRCN(ctx: click.Context, **kwargs: Any) -> None:
    kwargs.update(ctx.obj['common'])

    train_dl, valid_dl = ctx.obj['train_dl'], ctx.obj['valid_dl']

    n_classes = len(train_dl.dataset.classes)

    if kwargs['audio']:
        model, checkpoint = _load_model('LRCNWithAudio', 
                                        n_classes, 
                                        kwargs['feature_extractor'], 
                                        kwargs['freeze_fe'],
                                        kwargs['resume_checkpoint'],
                                        rnn_units=kwargs['rnn_units'], 
                                        bidirectional=kwargs['bidirectional'],
                                        audio_features=kwargs['audio_features'],
                                        n_mel_features=kwargs['mel_features'])
    else:
        model, checkpoint = _load_model('LRCN', 
                                        n_classes, 
                                        kwargs['feature_extractor'], 
                                        kwargs['freeze_fe'],
                                        kwargs['resume_checkpoint'],
                                        rnn_units=kwargs['rnn_units'], 
                                        bidirectional=kwargs['bidirectional'])
    model.to(device)

    train(model, train_dl, valid_dl, train_from=checkpoint, **kwargs)


if __name__ == "__main__":
    main()