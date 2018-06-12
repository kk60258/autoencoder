import sys
from six.moves import urllib
import tarfile

IMAGE_SIZE = 24

# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
datadir = '/tmp'

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

def get_train_inputs(data_dir=datadir):

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
    dataset = distorted_inputs(data_dir=data_dir)
    return dataset

def get_test_inputs(data_dir=datadir):
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
    dataset = distorted_test_inputs(data_dir=data_dir)
    return dataset


import os

import tensorflow as tf



def distorted_inputs(data_dir=datadir):
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

    return __get_dataset(filenames, augmentation=True)

def distorted_test_inputs(data_dir=datadir):
    """Construct distorted input for CIFAR training using the Reader ops.
    Args:
      data_dir: Path to the CIFAR-10 data directory.
      batch_size: Number of images per batch.
    Returns:
      images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
      labels: Labels. 1D tensor of [batch_size] size.
    """
    filenames = [os.path.join(data_dir, 'test_batch.bin')]

    return __get_dataset(filenames)


def __get_dataset(filenames, augmentation=False):
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
        image = tf.image.resize_images(image, [new_height, new_width])
        # image_data = tf.image.convert_image_dtype(image_data, dtype=tf.uint8)
        image = tf.cast(image, dtype=tf.float32)

        if augmentation:

            # Image processing for training the network. Note the many random
            # distortions applied to the image.

            # Randomly crop a [height, width] section of the image.
            distorted_image = tf.random_crop(image, [height, width, 3])

            # Randomly flip the image horizontally.
            distorted_image = tf.image.random_flip_left_right(distorted_image)

            # Because these operations are not commutative, consider randomizing
            # the order their operation.
            # NOTE: since per_image_standardization zeros the mean and makes
            # the stddev unit, this likely has no effect see tensorflow#1458.
            distorted_image = tf.image.random_brightness(distorted_image,
                                                         max_delta=63)
            distorted_image = tf.image.random_contrast(distorted_image,
                                                       lower=0.2, upper=1.8)
            image = distorted_image

        image = tf.image.per_image_standardization(image)

        return image, label

    dataset = dataset.map(transform)

    return dataset

maybe_download_and_extract()