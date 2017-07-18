from PIL import Image
import numpy as np

class ImagePreparer(object):
    def __init__(self, img_size, conv_to_grayscale=False):
        # img_size is a 3-tuple of form (height, width, depth)
        self.make_grayscale = conv_to_grayscale
        self.img_width = img_size[1]
        self.img_height = img_size[0]
        self.num_dims = 2
        if len(img_size) == 3 and not conv_to_grayscale:
            self.img_depth = img_size[2]
            self.num_dims = 3
        elif len(img_size) > 3:
            raise TooManyDimensionsException("ImagePreparer cannot create image with more than 3 dimensions")

    def conv_img_to_arr(self, img_path):
        im = Image.open(img_path)
        w, h = im.size
        if w > self.img_width:
            im = im.resize((self.img_width, int(h * self.img_width/w)))
            w, h = im.size
        if h > self.img_height:
            im = im.resize((int(w * self.img_height/h), self.img_height))
            w, h = im.size
        if self.make_grayscale:
            im = im.convert(mode="L")

        im_w_bg = Image.new(im.mode, (self.img_width, self.img_height))
        offset = (self.img_width - w)//2, (self.img_height - h)//2
        im_w_bg.paste(im, offset)
        im_w_bg.save("test.png")
        # fin_shape = (h, w, self.img_depth) if self.num_dims == 3 else (h, w)
        # im_data = np.array(im.getdata()).reshape(fin_shape)
        #
        # print(im_data.shape)


    def synthesize_new_data(self, img_arr):
        pass


class TooManyDimensionsException(Exception):
    pass


if __name__ == "__main__":
    ip = ImagePreparer((100, 150, 3))

    ip.conv_img_to_arr("data/tank/00023.png")