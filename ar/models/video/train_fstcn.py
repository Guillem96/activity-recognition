from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import click
import torch
import torch.utils.data as data
import torchvision.transforms as T
from torchvision.datasets.samplers import RandomClipSampler
from torchvision.datasets.samplers import UniformClipSampler

import ar
import ar.transforms as VT
from ar.models.video.models import LRCN
from ar.models.video.train_utils import default_collate_fn
from ar.models.video.train_utils import load_datasets
from ar.models.video.train_utils import load_optimizer
from ar.typing import PathLike
from ar.typing import Scheduler

_AVAILABLE_DATASETS = {'kinetics400', 'UCF-101'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_preparation(
        dataset: str,
        *,
        data_dir: PathLike,
        frames_per_clip: int,
        writer: ar.typing.TensorBoard,
        batch_size: int,
        annotations_path: Optional[PathLike] = None,
        steps_between_clips: int = 1,
        workers: int = 1,
        validation_size: float = .1) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    Loads the datasets with corresponding transformations and creates two data
    loaders, one for train and validation
    """

    train_tfms = T.Compose([
        VT.VideoToTensor(),
        VT.VideoResize((128, 171)),
        VT.VideoRandomCrop((112, 112)),
        VT.VideoRandomHorizontalFlip(),
        VT.VideoNormalize(**VT.imagenet_stats),
        # VT.VideoRandomErase(scale=(0.02, 0.15))
    ])

    valid_tfms = T.Compose([
        VT.VideoToTensor(),
        VT.VideoResize((128, 171)),
        VT.VideoCenterCrop((112, 112)),
        VT.VideoNormalize(**VT.imagenet_stats),
    ])

    train_ds, valid_ds = load_datasets(dataset_type=dataset,
                                       root=data_dir,
                                       annotations_path=annotations_path,
                                       frames_per_clip=frames_per_clip,
                                       steps_between_clips=steps_between_clips,
                                       workers=workers,
                                       validation_size=validation_size,
                                       train_transforms=train_tfms,
                                       valid_transforms=valid_tfms)

    ar.logger.log_random_videos(train_ds,
                                writer=writer,
                                unnormalize_videos=True,
                                video_format='CTHW')

    if hasattr(train_ds, 'video_clips'):
        train_sampler = RandomClipSampler(train_ds.video_clips, 10)
        valid_sampler = UniformClipSampler(valid_ds.video_clips, 10)
    else:
        raise ValueError('Video dataset must have the video_clips attribute')

    train_dl = data.DataLoader(train_ds,
                               batch_size=batch_size,
                               num_workers=workers,
                               sampler=train_sampler,
                               collate_fn=default_collate_fn,
                               pin_memory=True)

    valid_dl = data.DataLoader(valid_ds,
                               batch_size=batch_size,
                               num_workers=workers,
                               sampler=valid_sampler,
                               collate_fn=default_collate_fn,
                               pin_memory=True)

    return train_dl, valid_dl


def train(model: ar.checkpoint.SerializableModule,
          optimizer: torch.optim.Optimizer,
          train_dl: data.DataLoader,
          valid_dl: data.DataLoader,
          *,
          epochs: int,
          grad_accum_steps: int,
          train_from: dict,
          fp16: bool,
          save_checkpoint: PathLike,
          summary_writer: Optional[ar.typing.TensorBoard],
          scheduler: Optional[Scheduler] = None) -> Dict[str, float]:
    """
    Trains the models along specified epochs with the given train and validation
    dataloader.
    """
    criterion_fn = torch.nn.NLLLoss()
    starting_epoch = train_from.get('epoch', -1) + 1

    metrics = [
        ar.metrics.accuracy, ar.metrics.top_3_accuracy,
        ar.metrics.top_5_accuracy
    ]

    for epoch in range(starting_epoch, epochs):
        ar.engine.train_one_epoch(dl=train_dl,
                                  model=model,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  loss_fn=criterion_fn,
                                  metrics=metrics,
                                  grad_accum_steps=grad_accum_steps,
                                  mixed_precision=fp16,
                                  epoch=epoch,
                                  summary_writer=summary_writer,
                                  device=device)

        eval_metrics = ar.engine.evaluate(dl=valid_dl,
                                          model=model,
                                          metrics=metrics,
                                          loss_fn=criterion_fn,
                                          epoch=epoch,
                                          summary_writer=summary_writer,
                                          mixed_precision=fp16,
                                          device=device)

        # Save the model jointly with the optimizer
        model.save(
            save_checkpoint,
            epoch=epoch,
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict() if scheduler is not None else {})

    return eval_metrics


def _load_model(out_units: int,
                feature_extractor: str,
                freeze_fe: bool,
                resume_checkpoint: Optional[PathLike] = None,
                **kwargs: Any) -> Tuple[ar.checkpoint.SerializableModule, dict]:

    if resume_checkpoint is None:
        checkpoint: dict = dict()
        model = LRCN(feature_extractor,
                     out_units,
                     freeze_feature_extractor=freeze_fe,
                     **kwargs)
    else:
        model, checkpoint = LRCN.load(resume_checkpoint,
                                      map_location=device,
                                      freeze_feature_extractor=freeze_fe)

    return model, checkpoint


@click.command()
# Data options
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
@click.option('--validation-split', type=float, default=.1)
@click.option('--data-loader-workers', type=int, default=2)
@click.option('--frames-per-clip', type=int, required=True)
@click.option('--clips-stride', type=int, default=1)
# Training behavior
@click.option('--epochs', type=int, default=10)
@click.option('--batch-size', type=int, default=16)
@click.option('--optimizer',
              type=click.Choice(['SGD', 'Adam', 'AdamW']),
              default='SGD')
@click.option('--grad-accum-steps', type=int, default=1)
@click.option('--learning-rate', type=float, default=1e-3)
@click.option('--scheduler',
              type=click.Choice(['OneCycle', 'Step', 'None']),
              default='None')
# Training optimizations
@click.option('--fp16/--no-fp16',
              default=False,
              help='Perform the forward pass of the model and the loss '
              'computation with mixed precision. The backward pass stays'
              'with fp32')
# Logging options
@click.option('--logdir',
              type=str,
              default=None,
              help='Directory for tensorboard logs. If set to None no logs will'
              ' be generated')
# Checkpointing options
@click.option('--resume-checkpoint',
              type=click.Path(dir_okay=False),
              default=None,
              help='Resume training from')
@click.option('--save-checkpoint',
              type=click.Path(dir_okay=False),
              default='models/model.pt',
              help='File to save the checkpoint')
@click.option('--feature-extractor',
              type=click.Choice(list(ar.nn._FEATURE_EXTRACTORS)),
              default='resnet18')
@click.option('--freeze-fe/--no-freeze-fe',
              default=False,
              help='Wether or not to fine tune the pretrained'
              ' feature extractor')
@click.option('--st',
              type=int,
              default=5,
              help='Stride to sample frames from video clips')
@click.option('--dt',
              type=int,
              default=9,
              help='Offset to create the Vdiff tensors')
@click.option('--scl-features',
              type=int,
              default=64,
              help='Features for the SCL branch')
@click.option('--tcl-features',
              type=int,
              default=64,
              help='Features for the TCL branch')
def main(dataset: str, data_dir: PathLike, annots_dir: PathLike,
         validation_split: float, data_loader_workers: int,
         frames_per_clip: int, clips_stride: int, epochs: int, batch_size: int,
         optimizer: str, grad_accum_steps: int, learning_rate: float,
         scheduler: str, fp16: bool, logdir: Optional[PathLike],
         resume_checkpoint: PathLike, save_checkpoint: PathLike,
         feature_extractor: str, freeze_fe: bool, st: int, dt: int,
         scl_features: int, tcl_features: int) -> None:
    ar.engine.seed()

    if logdir:
        summary_writer = ar.logger.build_summary_writter(logdir)
    else:
        summary_writer = None

    # Make the frames per clips large enough to have enough frames to generate
    # the strided clips and to compute the Vdiff
    frames_per_clip = (frames_per_clip + dt) * st

    train_dl, valid_dl = data_preparation(dataset,
                                          data_dir=data_dir,
                                          frames_per_clip=frames_per_clip,
                                          batch_size=batch_size,
                                          annotations_path=annots_dir,
                                          steps_between_clips=clips_stride,
                                          workers=data_loader_workers,
                                          validation_size=validation_split,
                                          writer=summary_writer)

    n_classes = len(train_dl.dataset.classes)

    model, checkpoint = _load_model(n_classes,
                                    feature_extractor=feature_extractor,
                                    freeze_fe=freeze_fe,
                                    resume_checkpoint=resume_checkpoint,
                                    st=st,
                                    dt=dt,
                                    scl_features=scl_features,
                                    tcl_features=tcl_features)
    model.to(device)

    torch_optimizer, torch_scheduler = load_optimizer(
        model,
        optimizer,
        scheduler,
        checkpoint=checkpoint,
        learning_rate=learning_rate,
        epochs=epochs,
        steps_per_epoch=len(train_dl))

    eval_metrics = train(model,
                         torch_optimizer,
                         train_dl,
                         valid_dl,
                         epochs=epochs,
                         grad_accum_steps=grad_accum_steps,
                         scheduler=torch_scheduler,
                         fp16=fp16,
                         summary_writer=summary_writer,
                         train_from=checkpoint,
                         save_checkpoint=save_checkpoint)

    if summary_writer is not None:
        hparams = {
            **model.config(), 'optimizer': optimizer,
            'learning_rate': learning_rate,
            'grad_accum_steps': grad_accum_steps,
            'scheduler': scheduler,
            'epochs': epochs,
            'batch_size': batch_size,
            'clips_stride': clips_stride,
            'frames_per_clip': frames_per_clip,
            'st': st,
            'dt': dt,
            'tcl_features': tcl_features,
            'scl_features': scl_features
        }

        summary_writer.add_hparams(hparams, eval_metrics)


if __name__ == "__main__":
    main()
