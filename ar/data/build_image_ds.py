import click
from pathlib import Path
from typing import Collection, Mapping

from google_images_download import google_images_download


def _build_folder_structure(out_path: Path, 
                            actions: Collection[str]) -> Mapping[str, Path]:
    action_2_path = dict()
    for a in actions:
        action_path = out_path / a
        action_path.mkdir(exist_ok=True, parents=True)
        action_2_path[a] = action_2_path
    
    return action_2_path


def _download_images(query: str, 
                     query_2_path: Mapping[str, Path], 
                     image_per_query: int): 

    response = google_images_download.googleimagesdownload()  

    arguments = {"keywords": query, 
                 "format": "jpg", 
                 "limit": image_per_query,
                 "output_directory": str(query_2_path[query])} 
    try: 
        response.download(arguments) 
      
    # Handling File NotFound Error
    except FileNotFoundError:  
        print('Error downloading image')

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
        _download_images(q, query_2_path, images_per_action)

if __name__ == "__main__":
    main()