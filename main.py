import tensorflow as tf
import numpy as np
from image_preparer import ImagePreparer, DataManager



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

    # Set up network
    x = tf.placeholder(tf.float32, shape=[None, 100, 150, 3], name="in_images")
    y = tf.placeholder(tf.float32, shape=[None, 4], name="labels")
    logits = network_fn(x)

    with tf.name_scope("xent"):
        xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
        tf.summary.scalar("xent", xent)
    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(1e-4)
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

    # Initialize filewriter and model saver
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("tensorboard/vehicle_net/1")
    writer.add_graph(sess.graph)
    merged_summary = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=5, name="VehicleNet")

    # Train
    batch_size = 100
    ip = ImagePreparer((100, 150))
    dm = DataManager("data", ip, exclude_folders=["videos"])
    for i in range(5000):
        x, y = dm.get_batch(i*batch_size, batch_size)
        

if __name__ == "__main__":
    tf.app.run()
