import click

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

import torchvision
import torchvision.transforms as T

from .classifier import ImageClassifier
from ar.transforms import imagenet_stats

from ar.metrics import accuracy
from ar.utils.nn import _FEATURE_EXTRACTORS
from ar.utils.engine import train_one_epoch, evaluate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(**kwargs):
    train_tfms = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(**imagenet_stats)
    ])

    valid_tfms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(**imagenet_stats)
    ])

    train_ds = torchvision.datasets.ImageFolder(kwargs['data_dir'], 
                                                transform=train_tfms)
    print(train_ds.class_to_idx)

    valid_ds = torchvision.datasets.ImageFolder(kwargs['data_dir'], 
                                                transform=valid_tfms)
    print(valid_ds.class_to_idx)

    train_len = int(len(train_ds) * (1 - kwargs['validation_split']))
    rand_idx = torch.randperm(len(train_ds))
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

    model = ImageClassifier(kwargs['feature_extractor'], 
                            len(valid_ds.dataset.classes))
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), 
                            lr=kwargs['learning_rate'])

    criterion_fn = torch.nn.NLLLoss()

    for epoch in range(kwargs['epochs']):
        train_one_epoch(dl=train_dl,
                        model=model,
                        optimizer=optimizer,
                        loss_fn=criterion_fn,
                        epoch=epoch,
                        device=device)
        
        evaluate(dl=valid_dl,
                 model=model,
                 metrics=[accuracy],
                 loss_fn=criterion_fn,
                 device=device)
        
        # Save the model jointly with the optimizer
        model.save(kwargs['save_checkpoint'], 
                   optimizer=optimizer.state_dict())


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
@click.option('--learning-rate', type=float, default=1e-3)

@click.option('--save-checkpoint', type=click.Path(dir_okay=False),
              default='models/model.pt', help='File to save the checkpoint')
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    main()
