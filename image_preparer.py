import numpy as np
from PIL import Image
from random import shuffle
import os

class ImagePreparer:
    def __init__(self, img_size, conv_to_grayscale=False):
        # img_size is a 3-tuple of form (height, width, depth) defining the input image dimensions
        self.make_grayscale = conv_to_grayscale
        self.img_width = img_size[1]
        self.img_height = img_size[0]
        self.img_depth = 1 if conv_to_grayscale else img_size[2]
        if len(img_size) != 3:
            raise Not3DimensionsException("ImagePreparer cannot create image with dimension not equal to 3")

    def conv_img_to_arr(self, im):
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
        im_data = np.array(im_w_bg.getdata()).reshape((self.img_height, self.img_width, self.img_depth))

        im_data = im_data/255
        return im_data

    def synthesize_new_data(self, im):
        # Run this synthesis prior to converting to an array
        # Flip left-to-right
        flipped_im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return flipped_im

class Not3DimensionsException(Exception):
    pass


class DataManager:
    def __init__(self):
        pass

    def import_data(self, classes, image_prep):
        x = []
        y = []
        logits_to_class = {}

        i = 0
        for word in classes:
            all_files = os.listdir("data/" + word)
            new_y_arr = np.zeros(len(classes))
            new_y_arr[i] = 1
            logits_to_class[word] = new_y_arr

            for file in all_files:
                img = Image.open(file)
                flip_img = image_prep.synthesize_new_data(img)
                img_arr = image_prep.conv_img_to_arr(img)
                flip_img_arr = image_prep.conv_img_to_arr(flip_img)

                x.append(img_arr)
                y.append(new_y_arr[:])
                x.append(flip_img_arr)
                y.append(new_y_arr[:])

            i += 1

        comb = list(zip(x, y))
        shuffle(comb)
        x[:], y[:] = zip(*comb)

        return x, y, logits_to_class


if __name__ == "__main__":
    ip = ImagePreparer((100, 150, 3))

    test_im = Image.open("data/airplane/00031.png")

    # im_arr = ip.conv_img_to_arr(test_im)
    ip.synthesize_new_data(test_im)