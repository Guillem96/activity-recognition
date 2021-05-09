import click

from ar.applications import generate_action_clip
from ar.applications import similarity_graph


@click.group()
def main() -> None:
    pass


main.add_command(generate_action_clip.main, name='generate-clips')
main.add_command(similarity_graph.main, name='build-similarity-graph')

if __name__ == "__main__":
    main()
