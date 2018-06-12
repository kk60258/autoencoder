""" Auto Encoder Example.
Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

import sys
from six.moves import urllib
import tarfile

class Cifar10data:


    IMAGE_SIZE = 24

    # Global constants describing the CIFAR-10 data set.
    NUM_CLASSES = 10
    NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
    NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
    datadir = '/tmp'

    def __init__(self):
        bindir = self.maybe_download_and_extract()
        self.dataset = self.distorted_inputs(bindir)
        self.dataset = self.dataset.batch(4).repeat()
        self.iterator = self.dataset.make_one_shot_iterator()
        self.next_batch = self.iterator.get_next()

        self.test_dataset = self.distorted_test_inputs(bindir)
        self.test_dataset = self.test_dataset.batch(4)
        self.test_iterator = self.test_dataset.make_initializable_iterator()
        self.test_next_batch = self.test_iterator.get_next()

    def next_train_batch(self, sess):
        train_images, train_labels = sess.run(self.next_batch)
        return train_images

    def next_test_batch(self, sess):
        test_images, test_labels = sess.run(self.test_next_batch)
        return test_images

    @staticmethod
    def maybe_download_and_extract(data_dir=datadir, DATA_URL=DATA_URL):
        """Download and extract the tarball from Alex's website."""
        dest_directory = data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                                 float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
            print()
            statinfo = os.stat(filepath)
            print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
        if not os.path.exists(extracted_dir_path):
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)
        return extracted_dir_path

    def get_train_inputs(self, data_dir=datadir):

        """Construct distorted input for CIFAR training using the Reader ops.
        Returns:
          images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        Raises:
          ValueError: If no data_dir
        """
        if not data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
        dataset = self.distorted_inputs(data_dir=data_dir)
        return dataset

    def get_test_inputs(self, data_dir=datadir):
        """Construct distorted input for CIFAR test using the Reader ops.
        Returns:
          images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        Raises:
          ValueError: If no data_dir
        """
        if not data_dir:
            raise ValueError('Please supply a data_dir')
        data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
        dataset = self.distorted_test_inputs(data_dir=data_dir)
        return dataset



    def distorted_inputs(self, data_dir):
        """Construct distorted input for CIFAR training using the Reader ops.
        Args:
          data_dir: Path to the CIFAR-10 data directory.
          batch_size: Number of images per batch.
        Returns:
          images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                     for i in range(1, 6)]

        return self.__get_dataset(filenames, augmentation=True)

    def distorted_test_inputs(self, data_dir):
        """Construct distorted input for CIFAR training using the Reader ops.
        Args:
          data_dir: Path to the CIFAR-10 data directory.
          batch_size: Number of images per batch.
        Returns:
          images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
          labels: Labels. 1D tensor of [batch_size] size.
        """
        filenames = [os.path.join(data_dir, 'test_batch.bin')]

        return self.__get_dataset(filenames)


    def __get_dataset(self, filenames, augmentation=False):
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
        # input format.
        label_bytes = 1  # 2 for CIFAR-100
        height = 32
        width = 32
        depth = 3
        image_bytes = height * width * depth
        # Every record consists of a label followed by the image, with a
        # fixed number of bytes for each.
        record_bytes = label_bytes + image_bytes

        new_height = 28
        new_width = 28

        dataset = tf.data.FixedLengthRecordDataset(filenames, record_bytes=record_bytes)

        def transform(value):
            # Convert from a string to a vector of uint8 that is record_bytes long.
            record_bytes = tf.decode_raw(value, tf.uint8)

            # The first bytes represent the label, which we convert from uint8->int32.
            label = tf.strided_slice(record_bytes, [0], [label_bytes])
            label = tf.cast(label, tf.uint8)
            label = tf.reshape(label, shape=[])
            label = tf.one_hot(label, depth=10)

            # label = tf.one_hot(label, depth=NUM_CLASSES)

            # The remaining bytes after the label represent the image, which we reshape
            # from [depth * height * width] to [depth, height, width].
            image = tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes])


            # Convert from [depth, height, width] to [height, width, depth].
            image = tf.reshape(image, [depth, height, width])
            image = tf.transpose(image, [1, 2, 0])

            ### resize image
            # image_file = tf.gfile.FastGFile(filenames[i], 'rb').read()
            # image = tf.image.decode_jpeg(image)
            image = tf.image.rgb_to_grayscale(image)
            image = tf.image.resize_images(image, [new_height, new_width])
            # image_data = tf.image.convert_image_dtype(image_data, dtype=tf.uint8)
            image = tf.cast(image, dtype=tf.float32)
            # image = tf.cast(image, dtype=tf.uint8)

            # if augmentation:
            #
            #     # Image processing for training the network. Note the many random
            #     # distortions applied to the image.
            #
            #     # Randomly crop a [height, width] section of the image.
            #     distorted_image = tf.random_crop(image, [height, width, 3])
            #
            #     # Randomly flip the image horizontally.
            #     distorted_image = tf.image.random_flip_left_right(distorted_image)
            #
            #     # Because these operations are not commutative, consider randomizing
            #     # the order their operation.
            #     # NOTE: since per_image_standardization zeros the mean and makes
            #     # the stddev unit, this likely has no effect see tensorflow#1458.
            #     distorted_image = tf.image.random_brightness(distorted_image,
            #                                                  max_delta=63)
            #     distorted_image = tf.image.random_contrast(distorted_image,
            #                                                lower=0.2, upper=1.8)
            #     image = distorted_image

            image = tf.image.per_image_standardization(image)

            return image, label

        dataset = dataset.map(transform)

        return dataset

TENSORBOARD_PATH = '/tmp/tensorboard/'
# Training Parameters
learning_rate = 0.01
num_steps = 1
batch_size = 32

display_step = 1000
examples_to_show = 10

# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)

tf.reset_default_graph()
# tf Graph input (only pictures)

mode = tf.placeholder(tf.bool)

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([num_hidden_2, num_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([num_input])),
}

def max_unpool_2x2(x, output_shape):
    out = tf.concat([x, tf.zeros_like(x)], 3)
    out = tf.concat([out, tf.zeros_like(out)], 2)
    out_size = output_shape
    return tf.reshape(out, out_size)

def max_pool_2x2(x):
    _, argmax = tf.nn.max_pool_with_argmax(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')
    pool = tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
    return pool, argmax

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Building the encoder
def encoder_cnn(layer):
    # Encoder Hidden layer with sigmoid activation #1
    layer = tf.layers.conv2d(inputs=layer, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu, kernel_regularizer=None, name='conv1')
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2, 2], strides=[2, 2], padding='same', name = 'max_pool')
    layer = tf.layers.conv2d(inputs=layer, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.relu, kernel_regularizer=None, name='conv2')
    layer = tf.layers.max_pooling2d(inputs=layer, pool_size=[2, 2], strides=[2, 2], padding='same', name = 'max_pool')
    layer = tf.layers.flatten(inputs=layer, name='flatten_c')
    layer = tf.layers.dense(inputs=layer, activation=tf.nn.relu, units=10, name='fc_e1')

    # layer = tf.layers.batch_normalization(inputs=x, training=mode)
    return layer


# Building the decoder
def decoder_cnn(layer):
    layer = tf.layers.dense(inputs=layer, activation=tf.nn.sigmoid, units=49, name='fc_d1')
    layer = tf.reshape(tensor=layer, shape=[-1, 7, 7, 1])
    layer = tf.image.resize_nearest_neighbor(images=layer, size=[layer.shape[1] * 2, layer.shape[2] * 2])
    layer = tf.layers.conv2d_transpose(inputs=layer, filters=32, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.sigmoid, name='conv_trans1')
    layer = tf.image.resize_nearest_neighbor(images=layer, size=[layer.shape[1] * 2, layer.shape[2] * 2])
    layer = tf.layers.conv2d_transpose(inputs=layer, filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same', activation=tf.nn.sigmoid, name='conv_trans2')
    layer = tf.layers.flatten(inputs=layer, name='flatten_d')
    layer = tf.layers.dense(inputs=layer, activation=tf.nn.sigmoid, units=784, name='fc_d2')
    layer = tf.reshape(tensor=layer, shape=Input_shape)
    return layer

class InputData:
    def __init__(self):
        # Import MNIST data
        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets("/tmp/data/", one_hot=True, source_url='http://yann.lecun.com/exdb/mnist/')

        self.Input_shape = [-1, 28, 28, 1]
        self.X = tf.placeholder(tf.float32, [None, 28, 28, 1])

    def next_train_batch(self, n=32):
        next, _ = self.mnist.train.next_batch(n)
        return np.reshape(next, (-1, 28, 28, 1))

    def next_test_batch(self, n=32):
        next, _ = self.mnist.train.next_batch(n)
        return np.reshape(next, (-1, 28, 28, 1))

input = InputData()
Input_shape = input.Input_shape

input_cifar = Cifar10data()

# Construct model
encoder_op = encoder_cnn(input.X)
decoder_op = decoder_cnn(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = input.X

# Define loss and optimizer, minimize the squared error
loss = tf.losses.mean_squared_error(y_pred, y_true)
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session


with tf.Session() as sess:

    # if tf.gfile.Exists(TENSORBOARD_PATH):
    #     tf.gfile.DeleteRecursively(TENSORBOARD_PATH)
    # tf.gfile.MakeDirs(TENSORBOARD_PATH)

    summary_writer = tf.summary.FileWriter(TENSORBOARD_PATH, sess.graph)

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x = input.next_train_batch(batch_size)

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={input.X: batch_x})
        # Display logs per step
        if i % display_step == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))

        loss_summary = tf.Summary()
        loss_summary.value.add(tag='loss', simple_value = l)
        summary_writer.add_summary(loss_summary, global_step=i)

    # Testing
    # Encode and decode images from test set and visualize their reconstruction.
    n = 4
    for i in range(n):
        # MNIST test set
        batch_x = input.next_test_batch(n)
        # Encode and decode the digit image
        tf.summary.image('original', batch_x, collections=['image'])
        tf.summary.image('reconstruct', decoder_op, collections=['image'])
        merge = tf.summary.merge_all(key='image')
        _, summary = sess.run([decoder_op, merge], feed_dict={input.X: batch_x})
        summary_writer.add_summary(summary)

    for i in range(n):
        # MNIST test set
        cifar_x = input_cifar.next_train_batch(sess)
        # Encode and decode the digit image
        tf.summary.image('original_cifar', cifar_x, collections=['image_cifar'])
        tf.summary.image('reconstruct_cifar', decoder_op, collections=['image_cifar'])
        merge_cifar = tf.summary.merge_all(key='image_cifar')
        _, summary_cifar = sess.run([decoder_op, merge_cifar], feed_dict={input.X: cifar_x})
        summary_writer.add_summary(summary_cifar)

    summary_writer.flush()