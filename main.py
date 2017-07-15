import tensorflow as tf
import numpy as np
import pickle

def import_data(path):
    with open(path, 'rb') as d_file:
        data = pickle.load(d_file, encoding="bytes")
        print(data)
    return data


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


def network_fn():
    input_layer = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])

    conv1 = conv_layer(input_layer, 3, 40, (5, 5), (2, 2), name="conv1")
    conv2 = conv_layer(conv1, 40, 60, (4, 4), (2, 2), name="conv2")
    conv3 = conv_layer(conv2, 60, 80, (3, 3), (2, 2), name="conv3")

    flat_conv3 = tf.reshape(conv3, [-1, 80*4*4], name="flatten")

    fcl1 = fc_layer(flat_conv3, 80*4*4, 2560, name="fc1")
    fcl2 = fc_layer(fcl1, 2560, 1024, name="fc2")

    logits = fc_layer(1024, 10, name="logits")

    return logits


def main(argv):
    sess = tf.Session()
    import_data("data/cifar-10-python.tar.gz")


if __name__ == "__main__":
    tf.app.run()
