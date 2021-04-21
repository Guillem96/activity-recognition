import uuid
import json
import shutil
import urllib
import subprocess
import multiprocessing
from pathlib import Path
from filelock import FileLock
from functools import partial
from typing import Optional, Union, Mapping, Any, Tuple

import click
import tqdm
import pandas as pd
from pytube import YouTube


def create_video_folders(
        dataset: pd.DataFrame, output_dir: Union[str, Path],
        tmp_dir: Union[str, Path]) -> Union[Mapping[str, Path], Path]:
    """Creates a directory for each label name in the dataset."""
    if 'label-name' not in dataset.columns:
        this_dir = Path(output_dir, 'test')
        this_dir.mkdir(exist_ok=True, parents=True)
        return this_dir

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)

    label_to_dir = {}
    for label_name in dataset['label-name'].unique():
        this_dir = output_dir / label_name.replace(' ', '_')
        this_dir.mkdir(exist_ok=True)
        label_to_dir[label_name] = this_dir

    return label_to_dir


def construct_video_filename(
        row: Mapping[str, Any],
        label_to_dir: Union[str, Mapping[str, Path]]) -> Path:
    """Given a dataset row, this function constructs the
       output filename for a given video.
    """
    v_id = row['video-id']
    start_t, end_t = row.get('start-time'), row.get('end-time')

    if start_t is None or end_t is None:
        fname = f'{v_id}.mp4'
    else:
        fname = f'{v_id}_{start_t:06d}_{end_t:06d}.mp4'

    if isinstance(label_to_dir, str):
        dirname = Path(label_to_dir)
    else:
        dirname = label_to_dir[row['label-name']]

    return dirname / fname


def download_clip(video_identifier: str,
                  output_filename: Union[str, Path],
                  start_time: Optional[int] = None,
                  end_time: Optional[int] = None,
                  tmp_dir: Union[str, Path] = '/tmp/kinetics',
                  url_base: str = 'https://www.youtube.com/watch?v=') \
                      -> Tuple[bool, str]:
    """Download a video from youtube if exists and is not blocked.

    Parameters
    ----------
    video_identifier: str
        Unique YouTube video identifier (11 characters)
    output_filename: str
        File path where the video will be stored.
    start_time: float
        Indicates the begining time in seconds from where the video
        will be trimmed.
    end_time: float
        Indicates the ending time in seconds of the trimmed video.
    """
    # Construct command line for getting the direct video link.
    fname = uuid.uuid4()

    try:
        # TODO: Is there a faster library?
        yt = (YouTube(url_base + video_identifier).streams.filter(
            file_extension='mp4',
            progressive=True).get_lowest_resolution().download(
                output_path=str(tmp_dir), filename=str(fname)))
    except urllib.error.HTTPError as err:  # type: ignore[attr-defined]
        if err.code == 429:
            print('Stopping downloading because we reached '
                  'the maximum requests per day')

        print('Error', str(err))
        return False, str(err.code)

    except Exception as err:
        print('Error', str(err))
        return False, str(err)

    tmp_filename = Path(tmp_dir, str(fname) + '.mp4')

    if start_time is None and end_time is None:
        shutil.copy(str(tmp_filename), str(output_filename))
    else:
        command = ' '.join([
            'ffmpeg',
            '-i',
            f'"{str(tmp_filename)}"',
            '-ss',
            str(start_time),
            '-t',
            str(end_time - start_time),
            '-c:v',
            'libx264',
            '-c:a',
            'copy',
            '-threads',
            '1',
            # '-loglevel', 'panic',
            f'"{str(output_filename)}"'
        ])
        try:
            output = subprocess.check_output(command,
                                             shell=True,
                                             stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as err:
            # print(f'[ERROR] Croping video {tmp_filename}')
            return False, err.output.decode('utf-8')

    # Check if the video was successfully saved.
    status = tmp_filename.exists()
    if status:
        tmp_filename.unlink()
    return status, 'Downloaded'


def download_clip_wrapper(row: Mapping[str, Any], label_to_dir: Mapping[str,
                                                                        Path],
                          tmp_dir: Union[str,
                                         Path], queue: multiprocessing.Queue,
                          fail_file: Union[str, Path]) -> Tuple[bool, str]:
    """Wrapper for parallel processing purposes."""
    with FileLock(str(fail_file) + '.lock'):
        with Path(fail_file).open() as f:
            register = json.load(f)
            if row['video-id'] in register:
                return False, 'Error already registered'

    output_filename = construct_video_filename(row, label_to_dir)
    if output_filename.exists():
        return True, 'Fail already exists'

    downloaded, log = download_clip(row['video-id'],
                                    output_filename,
                                    row.get('start-time'),
                                    row.get('end-time'),
                                    tmp_dir=tmp_dir)

    if not downloaded:
        queue.put((row['video-id'], log))

    return downloaded, str(log)


def parse_kinetics_annotations(input_csv: str,
                               ignore_is_cc: bool = False) -> pd.DataFrame:
    """Returns a parsed DataFrame.

    arguments:
    ---------
    input_csv: str
        Path to CSV file containing the following columns:
          'YouTube Identifier,Start time,End time,Class label'

    returns:
    -------
    dataset: DataFrame
        Pandas with the following columns:
            'video-id', 'start-time', 'end-time', 'label-name'
    """
    df = pd.read_csv(input_csv)
    if 'youtube_id' in df.columns:
        columns = {
            'youtube_id': 'video-id',
            'time_start': 'start-time',
            'time_end': 'end-time',
            'label': 'label-name'
        }

        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]

    return df


def writer(q: multiprocessing.Queue, fail_file: Union[Path, str]) -> None:
    """Process to log the failures"""

    fail_file = Path(fail_file)

    while 1:
        recv = q.get()
        if isinstance(recv, tuple):
            with FileLock(str(fail_file) + '.lock'):
                video_id, log = recv
                with fail_file.open('r') as f:
                    register = json.load(f)
                    register[video_id] = log

            with FileLock(str(fail_file) + '.lock'):
                with fail_file.open('w') as f:
                    json.dump(register, f)
                    f.flush()

        elif isinstance(recv, str):
            if recv == 'done': return


@click.command()
@click.argument(
    'input_csv',
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    # help='CSV file containing the following format: '
    #      'YouTube Identifier,Start time,End time,Class label'
)
@click.argument(
    'output_dir',
    required=True,
    type=click.Path(file_okay=False),
    # help='Output directory where videos will be saved.'
)
@click.option('--failed-file',
              '-f',
              default='.failed.json',
              type=click.Path(dir_okay=False),
              help='File to keep track of failed download and not repeating '
              'them every time we resume the downloading script')
@click.option('--num-jobs', '-n', default=8, type=int)
@click.option('--tmp-dir',
              '-t',
              default='/tmp/kinetics',
              type=click.Path(file_okay=False))
def main(input_csv: str, output_dir: str, failed_file: Union[str, Path],
         num_jobs: int, tmp_dir: str) -> None:

    failed_file = Path(failed_file)
    if not failed_file.exists():
        failed_file.write_text('{}')

    # Reading and parsing Kinetics.
    dataset = parse_kinetics_annotations(input_csv)

    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, output_dir, tmp_dir)

    # Download all clips.
    pool = multiprocessing.Pool(num_jobs)
    manager = multiprocessing.Manager()
    queue = manager.Queue()
    pbar = tqdm.tqdm(total=dataset.shape[0])

    def update(args: Any) -> None:
        successful, log = args
        if not successful and log == '429':
            pool.terminate()

        pbar.update(1)
        pbar.refresh()

    try:
        pool.apply_async(writer, args=(queue, failed_file))

        jobs = []
        for i, row in dataset.iterrows():
            args = (row, label_to_dir, tmp_dir, queue, failed_file)
            j = pool.apply_async(download_clip_wrapper,
                                 args=args,
                                 callback=update)
            jobs.append(j)

        for j in jobs:
            j.get()
        queue.put('done')

    except KeyboardInterrupt:
        print('Stopped downloads pressing CTRL + C')
        pool.terminate()
    else:
        pool.close()

    pool.join()


if __name__ == '__main__':
    main()
