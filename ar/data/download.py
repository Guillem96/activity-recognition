import uuid
import json
import shutil
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from collections import OrderedDict
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
                  start_time: int, end_time: int,
                  tmp_dir: str = '/tmp/kinetics',
                  num_attempts: int = 5,
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
    except Exception as err:
        # print('[ERROR]', err)
        return False, str(err)

    # Construct command to trim the videos (ffmpeg required).
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
    tmp_filename.unlink(missing_ok=True)
    return status, 'Downloaded'


def download_clip_wrapper(row: Mapping[str, Any], 
                          label_to_dir: Mapping[str, Path],  
                          tmp_dir: Union[str, Path]):
    """Wrapper for parallel processing purposes."""
    output_filename = construct_video_filename(
        row, label_to_dir)

    clip_id = output_filename.stem
    if output_filename.exists():
        return clip_id, True, 'Exists'

    downloaded, log = download_clip(row['video-id'], output_filename,
                                    row['start-time'], row['end-time'],
                                    tmp_dir=tmp_dir)
    return clip_id, downloaded, str(log)


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
        columns = OrderedDict([
            ('youtube_id', 'video-id'),
            ('time_start', 'start-time'),
            ('time_end', 'end-time'),
            ('label', 'label-name')])
        df.rename(columns=columns, inplace=True)
        if ignore_is_cc:
            df = df.loc[:, df.columns.tolist()[:-1]]

    return df


def main(input_csv: str, 
         output_dir: str,
         num_jobs: int = 24, 
         tmp_dir: str = '/tmp/kinetics'):

    # Reading and parsing Kinetics.
    dataset = parse_kinetics_annotations(input_csv)

    # Creates folders where videos will be saved later.
    label_to_dir = create_video_folders(dataset, output_dir, tmp_dir)


    # Download all clips.
    if num_jobs == 1:
        status_lst = []
        for i, row in tqdm.tqdm(dataset.iterrows(), total=dataset.shape[0]):
            status_lst.append(download_clip_wrapper(row, label_to_dir,
                                                    tmp_dir))
    else:
        pool = multiprocessing.Pool(num_jobs)
        pbar = tqdm.tqdm(total=dataset.shape[0])
        def update(*args):
            pbar.update(1)
            pbar.refresh()

        try:
            for i, row in dataset.iterrows():
                pool.apply_async(download_clip_wrapper, 
                                 args=(row, label_to_dir, tmp_dir),
                                 callback=update)
        except KeyboardInterrupt:
            print('Stopped downloads pressing CTRL + C')
            pool.terminate()
        else:
            pool.close()
        
        pool.join()

    # Clean tmp dir.
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    description = 'Helper script for downloading and trimming kinetics videos.'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('input_csv', type=str,
                   help=('CSV file containing the following format: '
                         'YouTube Identifier,Start time,End time,Class label'))
    p.add_argument('output_dir', type=str,
                   help='Output directory where videos will be saved.')
    p.add_argument('-n', '--num-jobs', type=int, default=8)
    p.add_argument('-t', '--tmp-dir', type=str, default='/tmp/kinetics')
    main(**vars(p.parse_args()))
