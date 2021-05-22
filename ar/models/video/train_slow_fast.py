from typing import Optional
from typing import Tuple

import click
import torch

import ar
from ar.models.video.models import SlowFast
from ar.models.video.train_utils import data_preparation
from ar.models.video.train_utils import load_optimizer
from ar.models.video.train_utils import train
from ar.typing import PathLike

_AVAILABLE_DATASETS = {'kinetics400', 'UCF-101'}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _load_model(
    out_units: int,
    alpha: float,
    beta: float,
    tau: int,
    dropout: float,
    fusion_mode: str,
    resume_checkpoint: Optional[PathLike] = None,
) -> Tuple[ar.SerializableModule, dict]:

    if resume_checkpoint is None:
        checkpoint: dict = dict()
        model = SlowFast(out_units,
                         alpha=alpha,
                         beta=beta,
                         tau=tau,
                         dropout=dropout,
                         fusion_mode=fusion_mode)
    else:
        model, checkpoint = SlowFast.load(resume_checkpoint,
                                          map_location=device)

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
@click.option('--base-fps', type=int, default=30)
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
@click.option('--alpha', type=float, default=8)
@click.option('--beta', type=float, default=1 / 8)
@click.option('--tau', type=int, default=16)
@click.option('--dropout', type=float, default=.3)
@click.option('--fusion-mode',
              type=click.Choice([
                  'time-to-channel', 'time-strided-sample', 'time-strided-conv',
              ]),
              default='time-strided-conv')
def main(
    dataset: str,
    data_dir: PathLike,
    annots_dir: PathLike,
    validation_split: float,
    data_loader_workers: int,
    base_fps: int,
    clips_stride: int,
    epochs: int,
    batch_size: int,
    optimizer: str,
    grad_accum_steps: int,
    learning_rate: float,
    scheduler: str,
    fp16: bool,
    logdir: Optional[PathLike],
    resume_checkpoint: PathLike,
    save_checkpoint: PathLike,
    alpha: float,
    beta: float,
    tau: int,
    dropout: float,
    fusion_mode: str,
) -> None:
    ar.engine.seed()

    if logdir:
        summary_writer = ar.logger.build_summary_writter(logdir)
    else:
        summary_writer = None

    desired_slow_frames = 4
    frames_per_clip = desired_slow_frames * tau
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
                                    alpha=alpha,
                                    beta=beta,
                                    tau=tau,
                                    dropout=dropout,
                                    resume_checkpoint=resume_checkpoint,
                                    fusion_mode=fusion_mode)
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
                         save_checkpoint=save_checkpoint,
                         device=device)

    if summary_writer is not None:
        hparams = {
            **model.config(), 'optimizer': optimizer,
            'learning_rate': learning_rate,
            'grad_accum_steps': grad_accum_steps,
            'scheduler': scheduler,
            'epochs': epochs,
            'batch_size': batch_size,
            'clips_stride': clips_stride,
            'frames_per_clip': frames_per_clip
        }

        summary_writer.add_hparams(hparams, eval_metrics)


if __name__ == "__main__":
    main()