from script.ImageManager import ImageManager
import click
from os.path import splitext


@click.command()
@click.argument("input_dir", type=str)
@click.option("--output_dir", type=str)
def main(input_dir, output_dir):
    input_dir = input_dir.encode('ascii')
    if output_dir:
        output_dir = output_dir.encode('ascii')
    pattern = input_dir.encode('ascii').split('/')[-2]
    manager = ImageManager(input_dir, output_dir)
    manager.rename_images_with_pattern(pattern)


if __name__=="__main__":
    main()