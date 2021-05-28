import click

from ar.data.cli import build_image_ds
from ar.data.cli import build_video_candidates
from ar.data.cli import download

from .models.video import train_fstcn
from .models.video import train_lrcn
from .models.video import train_r2plus1d_18
from .models.video import train_slow_fast


@click.group()
def main() -> None:
    pass


main.add_command(build_image_ds.main, name='data-image-dataset')
main.add_command(build_video_candidates.main, name='data-video-candidates')
main.add_command(download.main, name='data-download-videos')

main.add_command(train_lrcn.main, 'train-lrcn')
main.add_command(train_fstcn.main, 'train-fstcn')
main.add_command(train_r2plus1d_18.main, 'train-r2plus1d')
main.add_command(train_slow_fast.main, 'train-slowfast')

if __name__ == "__main__":
    main()
