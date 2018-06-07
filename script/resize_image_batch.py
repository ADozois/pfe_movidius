from script.ImageManager import ImageManager
import click


@click.command()
@click.argument("input_dir", type=str)
@click.argument("output_dir", type=str)
def main(input_dir, output_dir):
    manager = ImageManager(input_dir.encode('ascii'), output_dir.encode('ascii'))
    manager.execute_resize(224, 224)


if __name__=="__main__":
    main()