import click

from .data import download
from .models.video import train_fstcn
from .models.video import train_lrcn
from .models.video import train_r2plus1d_18
from .models.video import train_slow_fast


@click.group()
def main() -> None:
    pass


main.add_command(train_lrcn.main, 'train-lrcn')
main.add_command(train_fstcn.main, 'train-fstcn')
main.add_command(train_r2plus1d_18.main, 'train-r2plus1d')
main.add_command(train_slow_fast.main, 'train-slowfast')

if __name__ == "__main__":
    main()
