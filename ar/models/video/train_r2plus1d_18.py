import math
from typing import Any
from typing import Optional
from typing import Tuple

import accelerate
import click

import ar
from ar.models.video.models import R2plus1_18
from ar.models.video.train_utils import data_preparation
from ar.models.video.train_utils import load_optimizer

_AVAILABLE_DATASETS = {'kinetics400', 'UCF-101'}


def _load_model(out_units: int,
                freeze_fe: bool,
                resume_checkpoint: Optional[ar.typing.PathLike] = None,
                **kwargs: Any) -> Tuple[ar.SerializableModule, dict]:

    if resume_checkpoint is None:
        checkpoint: dict = dict()
        model = R2plus1_18(out_units,
                           freeze_feature_extractor=freeze_fe,
                           **kwargs)
    else:
        model, checkpoint = R2plus1_18.load(resume_checkpoint,
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
@click.option('--cpu/--no-cpu', default=False, help='Force CPU?')
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
@click.option('--freeze-fe/--no-freeze-fe',
              default=False,
              help='Wether or not to fine tune the pretrained'
              ' feature extractor')
def main(dataset: str, data_dir: ar.typing.PathLike,
         annots_dir: ar.typing.PathLike, validation_split: float,
         data_loader_workers: int, frames_per_clip: int, clips_stride: int,
         epochs: int, batch_size: int, optimizer: str, grad_accum_steps: int,
         learning_rate: float, scheduler: str, fp16: bool, cpu: bool,
         logdir: Optional[ar.typing.PathLike],
         resume_checkpoint: ar.typing.PathLike,
         save_checkpoint: ar.typing.PathLike, freeze_fe: bool) -> None:
    ar.engine.seed()

    accelerator = accelerate.Accelerator(fp16=fp16, cpu=cpu)
    print("=== Accelerator State ===")
    print(accelerator.state)

    if logdir:
        summary_writer = ar.logger.build_summary_writter(logdir)
    else:
        summary_writer = None

    with ar.distributed.master_first(accelerator):
        train_ds, valid_ds = data_preparation(
            dataset,
            data_dir=data_dir,
            frames_per_clip=frames_per_clip,
            annotations_path=annots_dir,
            steps_between_clips=clips_stride,
            workers=data_loader_workers,
            validation_size=validation_split,
            writer=summary_writer,
            log_videos=accelerator.is_main_process)

    n_classes = len(train_ds.classes)

    with ar.distributed.master_first(accelerator):
        model, checkpoint = _load_model(n_classes,
                                        freeze_fe=freeze_fe,
                                        resume_checkpoint=resume_checkpoint)

    steps_per_epoch = math.ceil(len(train_ds) / batch_size / grad_accum_steps)
    torch_optimizer, torch_scheduler = load_optimizer(
        model,
        optimizer,
        scheduler,
        checkpoint=checkpoint,
        learning_rate=learning_rate,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch)

    eval_metrics = ar.engine.train(model,
                                   torch_optimizer,
                                   train_ds,
                                   valid_ds,
                                   accelerator,
                                   epochs=epochs,
                                   batch_size=batch_size,
                                   dl_workers=data_loader_workers,
                                   grad_accum_steps=grad_accum_steps,
                                   scheduler=torch_scheduler,
                                   summary_writer=summary_writer,
                                   train_from=checkpoint,
                                   save_checkpoint=save_checkpoint)

    if accelerator.is_main_process and summary_writer:
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


def run():
    main()


if __name__ == "__main__":
    main()
