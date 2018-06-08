from script.ImageManager import ImageManager
import click


@click.command()
@click.argument("input_dir", type=str)
@click.option("--output_dir", default=None, type=str)
def main(input_dir, output_dir):
    input_dir = input_dir.encode('ascii')
    if output_dir:
        output_dir = output_dir.encode('ascii')
    manager = ImageManager(input_dir, output_dir)
    manager.execute_resize(224, 224)


if __name__=="__main__":
    main()