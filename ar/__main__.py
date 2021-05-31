import click
import sys
import os
import inspect
import subprocess
from ar.data.cli import build_image_ds
from ar.data.cli import build_video_candidates
from ar.data.cli import download

from .models.video import train_fstcn
from .models.video import train_lrcn
from .models.video import train_r2plus1d_18
from .models.video import train_slow_fast


_data_commands_function = {
    'data-image-dataset': build_image_ds.main,
    'data-video-candidates': build_video_candidates.main,
    'data-download-videos': download.main,
}

_train_commands_function = {
  'train-lrcn': train_lrcn.main,
  'train-fstcn': train_fstcn.main,
  'train-r2plus1d': train_r2plus1d_18.main,
  'train-slowfast': train_slow_fast.main,
}

_train_commands_path = {k: os.path.abspath(inspect.getfile(v.callback)) 
                        for k, v in _train_commands_function.items()}

def main():
    if len(sys.argv) == 1:
        _help_message()
        return

    cmd = sys.argv[1]
    if cmd in {'--help', '-h'}:
        _help_message()
        return

    if len(sys.argv) >= 2 and cmd in _train_commands_path:
        fn = _train_commands_function[cmd]
        if len(sys.argv) == 2:
            _cmd_help(cmd, fn)
        elif len(sys.argv) == 3 and sys.argv[2] in {'--help', '-h'}:
            _cmd_help(cmd, fn)
        else:
            bash_cmd = ['accelerate', 'launch', 
                        _train_commands_path[cmd], *sys.argv[2:]]

            process = subprocess.Popen(bash_cmd)
            process.wait()
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                  returncode=process.returncode, cmd=cmd)

    elif len(sys.argv) >= 2 and cmd in _data_commands_function:
        fn = _data_commands_function[cmd]
        if len(sys.argv) == 2:
            _cmd_help(cmd, fn)
        else:
            fn()
    else:
        _help_message()


def _cmd_help(cmd_name: str, cmd: click.Command) -> None:
    print(f'=== Command "{cmd_name}" usage ===')
    with click.Context(cmd) as ctx:
        click.echo(cmd.get_help(ctx))


def _help_message() -> None:
    print("Usage: python -m ar [cmd] [--help|-h]")
    print("Available commands:")
    print('=== Train commands (Launched with accelerate) ===', end='\n\t- ')
    print("\n\t- ".join(_train_commands_path))

    print('=== Data commands ===', end='\n\t- ')
    print("\n\t- ".join(_data_commands_function))

    print('\n--help|-h: Prints this message')


if __name__ == "__main__":
    main()
