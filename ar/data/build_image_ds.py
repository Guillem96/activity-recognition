import re
import io
import time
import base64
import requests
from pathlib import Path
import multiprocessing as mp
from typing import Collection, Mapping, Union

from PIL import Image

from selenium import webdriver


def _build_folder_structure(out_path: Path, 
                            actions: Collection[str]) -> Mapping[str, Path]:
    action_2_path = dict()
    for a in actions:
        action_path = out_path / a
        action_path.mkdir(exist_ok=True, parents=True)
        action_2_path[a] = action_2_path
    
    return action_2_path


def _get_urls(driver, query: str) -> Collection[str]:
    url = f'https://www.google.com/search?q={query}&tbm=isch'

    driver.get(url)

    # Scroll to bottom three times in order to load as much images as possible
    for _ in range(4):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(1)

    # Get all images urls
    img_container = driver.find_element_by_class_name('mJxzWe')
    elems = img_container.find_elements_by_tag_name('img')
    return [e.get_property('src') for e in elems if e.get_property('src')]


def _download_single(i: int, url: str, out_dir: Path):
    
    if url.startswith('data'):
        url = re.sub('^data:image/.+;base64,', '', url)
        image_data = base64.b64decode(url)
    else:
        image_data = requests.get(url).content
    img = Image.open(io.BytesIO(image_data))
    img = img.convert('RGB')
    img.save(str(out_dir / f'{i}.jpg'), 'JPEG')


def _download_urls(urls: Collection[str], 
                   out_dir: Union[str, Path],
                   n_workers: int):
    
    out_dir = Path(out_dir)
    pool = mp.Pool(n_workers)
    
    futures = []
    for i, url in enumerate(urls):
        r = pool.apply_async(_download_single, args=(i, url, out_dir))
        futures.append(r)
    
    for f in futures:
        f.get()

    pool.close()
    pool.join()

    
def _download_images(query: str, 
                     query_2_path: Mapping[str, Path], 
                     image_per_query: int,
                     n_workers: int): 
    
    driver = webdriver.Firefox()
    urls = _get_urls(driver, query)
    driver.close()
    _download_urls(urls, str(query_2_path[query]), n_workers)
    
    
@click.command()
@click.option('--actions', type=click.Path(exists=True, dir_okay=False),
              required=True,
              help='File containing search queries (actions to download). '
                   'Each line will correspond to a search query or action')
@click.option('--out-path', type=click.Path(file_okay=False), required=True,
              help='Output path where images are going to be stored. '
                   'The images will be saved using the ImageNet format')
@click.option('--images-per-action', type=int, default=100)
@click.option('--num-workers', type=int, default=4, 
              help='Number of threads used to download the images')
def main(actions, out_path, images_per_action, num_workers):

    out_path = Path(out_path)

    class_names = Path(actions).read_text().split('\n')
    class_names = [o.strip() for o in class_names]

    query_2_path = _build_folder_structure(out_path, class_names)
    for q in class_names:
        _download_images(q, query_2_path, images_per_action, num_workers)

        
if __name__ == "__main__":
    main()
