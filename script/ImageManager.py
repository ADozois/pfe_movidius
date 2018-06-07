from os import listdir
from os.path import isfile, join, expanduser
from PIL import Image
import progressbar

class ImageManager:
    def __init__(self, images_path, output_dir=None):
        self._image_directory = expanduser(images_path)
        self.images = self.extract_images_list()
        self._number_image = len(self.images)
        if not output_dir:
            self._output_dir = self._image_directory
        else:
            self._output_dir = expanduser(output_dir)

    def extract_images_list(self):
        return self.get_images_list_dir(self._image_directory)

    @staticmethod
    def get_images_list_dir(path):
        return [f for f in listdir(path) if isfile(join(path, f))]

    @staticmethod
    def resize(image, height, width):
        return image.resize((width, height), Image.BILINEAR)

    def execute_resize(self, width, height, format="JPEG"):
        count = 0
        with progressbar.ProgressBar(max_value=self._number_image) as bar:
            for image in self.images:
                img = Image.open(self._image_directory + image)
                img = self.resize(img, height, width)
                img.save(self._output_dir + image)
                count += 1
                bar.update(count)