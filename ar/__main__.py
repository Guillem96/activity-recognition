import click

from .data import download


@click.group()
def main() -> None:
    pass


@main.command()
def applications():
    pass


main.add_command(download.main, name='download-kinetics')


if __name__ == "__main__":
    main()
