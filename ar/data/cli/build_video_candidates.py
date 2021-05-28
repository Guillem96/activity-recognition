import os
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import click
import tqdm.auto as tqdm
from googleapiclient.discovery import Resource
from googleapiclient.discovery import build

YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'


@click.command()
@click.option('--actions',
              type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='File containing search queries (actions to download). '
              'Each line will correspond to a search query or action')
@click.option('--yt-api-key',
              help='Google API key',
              default=os.getenv('YOUTUBE_API_KEY'))
@click.option('--videos-per-action',
              help='Number of candidates per class',
              default=2,
              type=int)
@click.option(
    '--output',
    type=str,
    required=True,
    help='Output to store the video candidate clips. Recommended .csv extension.'
)
def main(actions: str, yt_api_key: Optional[str], videos_per_action: int,
         output: Union[Path, str]) -> None:
    if yt_api_key is None:
        raise ValueError(
            'yt-api-key not provided, use the --yt-api-key or the env '
            'variable "YOUTUBE_API_KEY"')

    output = Path(output)
    class_names = Path(actions).read_text().split('\n')
    youtube = build(YOUTUBE_API_SERVICE_NAME,
                    YOUTUBE_API_VERSION,
                    developerKey=yt_api_key)

    candidates = {
        action: _get_video_ids(youtube, action, videos_per_action)
        for action in tqdm.tqdm(class_names)
    }

    csv_text = 'label-name,video-id\n'

    for action, candid in candidates.items():
        for v in candid:
            csv_text += f'{action},{v}\n'

    with output.open('w') as out_f:
        out_f.write(csv_text)


def _get_video_ids(youtube: Resource, query: str, max_videos: int) -> List[str]:
    search_response = youtube.search().list(q=query,
                                            part='id,snippet',
                                            type='video',
                                            videoDuration='short',
                                            maxResults=max_videos).execute()

    video_results = search_response.get('items', [])

    return [
        o['id']['videoId']
        for o in video_results
        if o['id']['kind'] == 'youtube#video'
    ]


if __name__ == '__main__':
    main()
