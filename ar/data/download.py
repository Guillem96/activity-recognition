import uuid
import json
import time
import click
import shutil
import requests
import subprocess
from functools import partial

import multiprocessing
from pathlib import Path
from typing import Union, Mapping, Any

import tqdm
import pandas as pd
from pytube import YouTube


def create_video_folders(
    dataset: pd.DataFrame, 
    output_dir: Union[str, Path], 
    tmp_dir: Union[str, Path]) -> Union[Mapping[str, Path], Path]:

    """Creates a directory for each label name in the dataset."""
    if 'label-name' not in dataset.columns:
        this_dir = Path(output_dir, 'test') 
        this_dir.mkdir(exist_ok=True)
        return this_dir
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True, parents=True)

    label_to_dir = {}
    for label_name in dataset['label-name'].unique():
        this_dir = output_dir / label_name.replace(' ', '_')
        this_dir.mkdir(exist_ok=True)
        label_to_dir[label_name] = this_dir

    return label_to_dir


def construct_video_filename(row: Mapping[str, Any], 
                             label_to_dir: Mapping[str, Path]) -> Path:
    """Given a dataset row, this function constructs the
       output filename for a given video.
    """
    v_id, start_t, end_t = row['video-id'], row['start-time'], row['end-time']
    basename = f'{v_id}_{start_t:06d}_{end_t:06d}.mp4'

    if not isinstance(label_to_dir, dict):
        dirname = label_to_dir
    else:
        dirname = label_to_dir[row['label-name']]

    return dirname / basename


def download_clip(video_identifier: str, 
                  output_filename: Union[str, Path],
                  start_time: int, 
                  end_time: int,
                  tmp_dir: Union[str, Path] = '/tmp/kinetics',
                  url_base: str = 'https://www.youtube.com/watch?v='):
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
        yt = (YouTube(url_base + video_identifier)
              .streams
              .filter(file_extension='mp4', progressive=True)
              .get_lowest_resolution()
              .download(output_path=str(tmp_dir), 
                        filename=str(fname)))
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 429:
            import sys
            print('Stopping downloading because we reached' 
                  'the maximum requests per day')
            sys.exit(1)
        print('Error', str(err))
        return False, str(err.response.status_code)

    except Exception as err:
        print('Error', str(err))
        return False, str(err)

    tmp_filename = Path(tmp_dir, str(fname) + '.mp4')
    command = ['ffmpeg',
               '-i', f'"{str(tmp_filename)}"',
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
            #    '-loglevel', 'panic',
               f'"{str(output_filename)}"']
    command = ' '.join(command)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        # print(f'[ERROR] Croping video {tmp_filename}')
        return False, err.output.decode('utf-8')

    # Check if the video was successfully saved.
    status = tmp_filename.exists()
    if status: tmp_filename.unlink()
    return status, 'Downloaded'


def download_clip_wrapper(row: Mapping[str, Any], 
                          label_to_dir: Mapping[str, Path],
                          tmp_dir: Union[str, Path],
                          queue: multiprocessing.Queue,
                          fail_file: Union[str, Path]):
    """Wrapper for parallel processing purposes."""
    
    with Path(fail_file).open() as f:
        register = json.load(f)
        if row['video-id'] in register:
            return

    output_filename = construct_video_filename(
        row, label_to_dir)

    clip_id = output_filename.stem
    if output_filename.exists():
        return 

    downloaded, log = download_clip(row['video-id'], output_filename,
                                    row['start-time'], row['end-time'],
                                    tmp_dir=tmp_dir)

    if not downloaded:
        queue.put((row['video-id'], log))

    # return clip_id, downloaded, str(log)


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
        columns = {'youtube_id': 'video-id',
                   'time_start': 'start-time',
                   'time_end': 'end-time',
                   'label': 'label-name'}

        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]

    return df


def writer(q: multiprocessing.Queue, fail_file: Union[Path, str]):
    fail_file = Path(fail_file)

    while 1:
        video_id, log = q.get()
        with fail_file.open('r') as f:
            register = json.load(f)
            register[video_id] = log
        
        with fail_file.open('w') as f:
            json.dump(register, f)
            f.flush()


@click.command()
@click.argument('input_csv', required=True, 
                type=click.Path(exists=True, dir_okay=False),
                # help='CSV file containing the following format: '
                #      'YouTube Identifier,Start time,End time,Class label'
                     )
@click.argument('output_dir', required=True, type=click.Path(file_okay=False),
                # help='Output directory where videos will be saved.'
                )
@click.option('--failed-file', '-f', default='.failed.json', 
              type=click.Path(dir_okay=False), 
              help='File to keep track of failed download and not repeating '
                   'them every time we resume the downloading script')
@click.option('--num-jobs', '-n', default=8, type=int)
@click.option('--tmp-dir', '-t', default='/tmp/kinetics', 
              type=click.Path(file_okay=False))

def main(input_csv: str, 
         output_dir: str,
         failed_file: str,
         num_jobs: int, 
         tmp_dir: str):

    failed_file = Path(failed_file)
    if not failed_file.exists():
        failed_file.write_text('{}')
        
    # Reading and parsing Kinetics.
    dataset = parse_kinetics_annotations(input_csv)

    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, output_dir, tmp_dir)

    # Download all clips.
    if num_jobs == 1:
        for i, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
            download_clip_wrapper(row, label_to_dir, tmp_dir, failed_file)
    else:
        pool = multiprocessing.Pool(num_jobs)
        manager = multiprocessing.Manager()
        queue = manager.Queue()

        pbar = tqdm.tqdm(total=dataset.shape[0])
        def update(*args):
            pbar.update(1)
            pbar.refresh()
        try:
            jobs = []
            print('Launching logger process')

            jobs.append(pool.apply_async(writer, args=(queue, failed_file)))

            print('Launching workers')
            for i, row in dataset.iterrows():
                args = (row, label_to_dir, tmp_dir, queue, failed_file)
                j = pool.apply_async(download_clip_wrapper, args=args,
                                     callback=update)
                jobs.append(j)
            
            for j in jobs:
                j.get()

        except KeyboardInterrupt:
            print('Stopped downloads pressing CTRL + C')
            pool.terminate()
        else:
            pool.close()
        
        pool.join()

    # Clean tmp dir.
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    main()
