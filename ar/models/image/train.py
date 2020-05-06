import click
from typing import Any, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
import torchvision.transforms as T

from .classifier import ImageClassifier
from ar.transforms import imagenet_stats

from ar import engine
from ar.metrics import accuracy
from ar.utils.nn import _FEATURE_EXTRACTORS


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(**kwargs: Any) -> None:
    engine.seed()
    
    train_tfms = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(**imagenet_stats),
        T.RandomErasing(),
    ])

    valid_tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(**imagenet_stats)
    ])

    train_ds = torchvision.datasets.ImageFolder(kwargs['data_dir'], 
                                                transform=train_tfms)
    valid_ds = torchvision.datasets.ImageFolder(kwargs['data_dir'], 
                                                transform=valid_tfms)

    train_len = int(len(train_ds) * (1 - kwargs['validation_split']))
    rand_idx = torch.randperm(len(train_ds)).tolist()
    train_ds = Subset(train_ds, rand_idx[:train_len])
    valid_ds = Subset(valid_ds, rand_idx[train_len:])

    train_dl = DataLoader(train_ds, 
                          batch_size=kwargs['batch_size'],
                          num_workers=kwargs['data_loader_workers'],
                          shuffle=True)

    valid_dl = DataLoader(valid_ds, 
                          batch_size=kwargs['batch_size'],
                          num_workers=kwargs['data_loader_workers'],
                          shuffle=True)

    if kwargs['resume_checkpoint'] is None:
        checkpoint = None
        model = ImageClassifier(kwargs['feature_extractor'], 
                                len(valid_ds.dataset.classes),
                                freeze_feature_extractor=kwargs['freeze_fe'])
    else:
        model, checkpoint = ImageClassifier.load(
            kwargs['resume_checkpoint'],
            map_location=device,
            freeze_feature_extractor=kwargs['freeze_fe'])
        
    model.to(device)
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer: optim.Optimizer = None
    if kwargs['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(trainable_params, 
                                lr=kwargs['learning_rate'])
    elif kwargs['optimizer'] == 'SGD':
        optimizer = optim.SGD(trainable_params, lr=kwargs['learning_rate'],
                              momentum=0.9, weight_decay=4e-5)
    elif kwargs['optimizer'] == 'Adam':
        optimizer = optim.Adam(trainable_params, lr=kwargs['learning_rate'])
    
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
    if kwargs['scheduler'] == 'OneCycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, kwargs['learning_rate'] * 10, 
            len(train_dl) * kwargs['epochs'])
    elif kwargs['scheduler'] == 'Step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, len(train_dl), .1)
    elif kwargs['scheduler'] == 'None':
        scheduler = None
    
    if checkpoint is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        
        if 'scheduler' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        
        starting_epoch = checkpoint.get('epoch', -1) + 1

    criterion_fn = torch.nn.NLLLoss()

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


@click.command()

@click.option('--data-dir', type=click.Path(exists=True, file_okay=False),
              required=True, help='Dataset formated as imagenet directories')
@click.option('--validation-split', type=float, default=.1)

@click.option('--epochs', type=int, default=10)
@click.option('--batch-size', type=int, default=16)
@click.option('--data-loader-workers', type=int, default=2)

@click.option('--feature-extractor', 
              type=click.Choice(list(_FEATURE_EXTRACTORS)),
              default='resnet18')
@click.option('--freeze-fe/--no-freeze-fe', default=False,
              help='Wether or not to fine tune the pretrained'
                   ' feature extractor')
@click.option('--optimizer', type=click.Choice(['SGD', 'Adam', 'AdamW']),
              default='SGD')
@click.option('--grad-accum-steps', type=int, default=1)
@click.option('--learning-rate', type=float, default=1e-3)
@click.option('--scheduler', type=click.Choice(['OneCycle', 'Step', 'None']),
              default='None')

@click.option('--print-freq', type=int, default=20, 
              help='Print training epoch progress every n steps')

@click.option('--resume-checkpoint', type=click.Path(dir_okay=False),
              default=None, help='Resume training from')
@click.option('--save-checkpoint', type=click.Path(dir_okay=False),
              default='models/model.pt', help='File to save the checkpoint')
def main(**kwargs: Any) -> None:
    train(**kwargs)


if __name__ == "__main__":
    main()
