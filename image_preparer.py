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
    def __init__(self, data_dir, image_preparer, exclude_folders=None):
        self.data_dir = data_dir
        all_classes = os.listdir(data_dir)
        for folder in exclude_folders:
            if folder in all_classes:
                all_classes.remove(folder)
        self.all_classes = all_classes

        i = 0
        class_list = []
        data_list = []
        for word in all_classes:
            img_files = [word + "/" + f for f in os.listdir(data_dir + "/" + word)]
            logit = np.zeros((len(img_files), len(all_classes)))
            logit[:, i] = 1
            class_list.extend(list(logit))
            data_list.extend(img_files)
            i += 1
        comb = list(zip(class_list, data_list))
        shuffle(comb)
        self.class_list, self.data_list = zip(*comb)

        self.image_preparer = image_preparer

    def get_batch(self, start_pos, batch_size):
        batch_size //= 2
        start_pos = start_pos % len(self.data_list)
        stop_pos = (start_pos + batch_size) % len(self.data_list)

        x_file = self.data_list[start_pos:stop_pos]
        y = np.array(self.class_list[start_pos:stop_pos]).repeat(2, axis=0)

        x = []
        for imf in x_file:
            im = Image.open("data/" + imf)
            fl_im = self.image_preparer.synthesize_new_data(im)
            im = self.image_preparer.conv_img_to_arr(im)
            fl_im = self.image_preparer.conv_img_to_arr(fl_im)
            x.append(im)
            x.append(fl_im)

        return x, y


if __name__ == "__main__":
    ip = ImagePreparer((100, 150, 3))

    dm = DataManager("data", ip, exclude_folders=["videos"])

    x, y = dm.get_batch(1000, 2)
