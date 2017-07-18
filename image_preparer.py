from PIL import Image
import numpy as np

class ImagePreparer(object):
    def __init__(self, img_size, conv_to_grayscale=False):
        self.make_grayscale = conv_to_grayscale
        self.img_width = img_size[0]
        self.img_width = img_size[1]
        self.num_dims = 2
        if len(img_size) == 3:
            self.img_depth = img_size[2]
            self.num_dims = 3
        elif len(img_size) > 3:
            raise TooManyDimensionsException("ImagePreparer cannot create image with more than 3 dimensions")

    def conv_img_to_arr(self, img_path):
        im = Image.open(img_path)
        data = np.array(im.getdata()).reshape((im.size[1], im.size[0], 3))

        

    def synthesize_new_data(self, img_arr):
        pass

class TooManyDimensionsException(Exception):
    pass


if __name__ == "__main__":
    ip = ImagePreparer((100, 150))

    ip.conv_img_to_arr("data/airplane/00000.png")