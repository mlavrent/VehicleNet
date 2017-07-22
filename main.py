import tensorflow as tf
import numpy as np
from image_preparer import ImagePreparer
from PIL import Image
from random import shuffle
import os


def import_data(classes, image_prep):
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


def conv_layer(input, channels_in, channels_out, filter_size, pool_size, name="conv"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([*filter_size, channels_in, channels_out], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[channels_out]))

        act = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME") + b)
        pool = tf.nn.max_pool(act, ksize=[1, *pool_size, 1], strides=[1, *pool_size, 1], padding="SAME")

        tf.summary.histogram("act", act)

        return pool


def fc_layer(input, channels_in, channels_out, name="fcl"):
    with tf.name_scope(name):
        W = tf.Variable(tf.truncated_normal([channels_in, channels_out], stddev=0.1))
        b = tf.Variable(tf.constant(0., shape=[channels_out]))

        act = tf.nn.relu(tf.matmul(input, W) + b)

        tf.summary.histogram("act", act)
        return act


def network_fn(x):
    img_2d = tf.reshape(x, [-1, 32, 32, 3], name="2d_reshape")

    conv1 = conv_layer(img_2d, 3, 40, (5, 5), (2, 2), name="conv1")
    conv2 = conv_layer(conv1, 40, 60, (4, 4), (2, 2), name="conv2")
    conv3 = conv_layer(conv2, 60, 80, (3, 3), (2, 2), name="conv3")

    flat_conv3 = tf.reshape(conv3, [-1, 80*4*4], name="flatten")

    fcl1 = fc_layer(flat_conv3, 80*4*4, 2560, name="fc1")
    fcl2 = fc_layer(fcl1, 2560, 1024, name="fc2")

    logits = fc_layer(1024, 10, name="logits")

    return logits


def main(argv):
    sess = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, 100, 150, 3], name="in_images")
    y = tf.placeholder(tf.float32, shape=[None, 4], name="labels")
    logits = network_fn(x)

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

if __name__ == "__main__":
    tf.app.run()
