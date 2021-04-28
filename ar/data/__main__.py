import click

from ar.data import build_image_ds
from ar.data import build_video_candidates
from ar.data import download

@click.group()
def main() -> None:
    pass


main.add_command(build_image_ds.main, name='image-dataset')
main.add_command(build_video_candidates.main, name='video-candidates')
main.add_command(download.main, name='download-videos')


if __name__ == "__main__":
    main()
