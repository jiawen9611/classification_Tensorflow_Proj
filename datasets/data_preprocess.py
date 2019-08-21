# -*-coding:utf-8-*-
import numpy as np
import pickle
import cv2
import os
import tarfile
import sys
import glob
import math
import tensorflow as tf
from six.moves import urllib

'''
for data reading and data augmentation
'''


# only for cifar10
def generate_vali_batch(vali_data, vali_label, vali_batch_size):
    offset = np.random.choice(10000 - vali_batch_size, 1)[0]
    vali_data_batch = vali_data[offset:offset + vali_batch_size, ...]
    vali_label_batch = vali_label[offset:offset + vali_batch_size]
    return vali_data_batch, vali_label_batch


def generate_augment_train_batch(train_data, train_labels, config):
    train_batch_size = config.batch_size
    if config.dataset == 'cifar10':
        offset = np.random.choice(50000 - train_batch_size, 1)[0]
        batch_data = train_data[offset:offset + train_batch_size, ...]
        batch_data = random_crop_and_flip(batch_data, config)
        # batch_data = whitening_image(batch_data, config)
        batch_label = train_labels[offset:offset + train_batch_size]
    elif config.dataset == 'captcha':
        indices = np.random.choice(len(train_labels), train_batch_size)
        batch_data = train_data[indices]
        batch_label = train_labels[indices]
    elif config.dataset == 'easy':
        indices = np.random.choice(len(train_labels), train_batch_size)
        batch_data = train_data[indices]
        batch_label = train_labels[indices]

    return batch_data, batch_label


def horizontal_flip(image, axis):
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        # careful !!! todo: this change the RGB???
        image = cv2.flip(image, axis)
    return image


def random_crop_and_flip(batch_data, config):
    padding_size = config.aug_padding
    IMG_HEIGHT = config.input_size_h
    IMG_WIDTH = config.input_size_w
    IMG_DEPTH = config.input_size_d

    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset + IMG_HEIGHT, y_offset: y_offset + IMG_WIDTH, :]
        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)
    return cropped_batch


def maybe_download_and_extract_cifar10():
    '''
    Will download and extract the cifar10 data automatically
    :return: nothing
    '''
    dest_directory = 'datasets/cifar10'
    DATA_URL = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size)
                                                             / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def read_train_data(config_dict=None):
    path_list = []

    if config_dict.dataset == 'cifar10':
        maybe_download_and_extract_cifar10()
        NUM_TRAIN_BATCH = 5
        for i in range(1, NUM_TRAIN_BATCH + 1):
            path_list.append(config_dict.data_path + 'cifar-10-batches-py/data_batch_' + str(i))
        data, label = read_images(config_dict, path_list, shuffle=True, is_random_label=False)
        # preprocess: padding
        pad_width = (
            (0, 0), (config_dict.aug_padding, config_dict.aug_padding),
            (config_dict.aug_padding, config_dict.aug_padding),
            (0, 0))
        data = np.pad(data, pad_width=pad_width, mode='constant', constant_values=0)

    elif config_dict.dataset == 'captcha':
        if not os.path.exists(config_dict.data_path):
            raise ValueError('images_path is not exist.')

        images = []
        labels = []
        images_path = os.path.join(config_dict.data_path, '*.jpg')
        count = 0
        for image_file in glob.glob(images_path):
            count += 1
            if count % 1000 == 0:
                print('Load {} images.'.format(count))
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Assume the name of each image is imagexxx_label.jpg
            label = int(image_file.split('_')[-1].split('.')[0])
            images.append(image)
            labels.append(label)
        data = np.array(images)
        label = np.array(labels)
    elif config_dict.dataset == 'easy':
        if not os.path.exists(config_dict.data_path):
            raise ValueError('images_path is not exist.')

        images = []
        labels = []
        images_path = os.path.join(config_dict.data_path, '*.jpg')
        count = 0
        for image_file in glob.glob(images_path):
            count += 1
            if count % 100 == 0:
                print('Load {} images.'.format(count))
            image = cv2.imread(image_file)
            image = cv2.resize(image, (config_dict.input_resize_w, config_dict.input_resize_h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Assume the name of each image is imagexxx_label.jpg
            label = int(ord(image_file.split('_')[-3].split('/')[-1]) - 65)
            images.append(image)
            labels.append(label)
        data = np.array(images)
        label = np.array(labels)

    return data, label


def read_validation_data(config_dict=None):
    path_list = []
    if config_dict.dataset == 'cifar10':
        path_list.append(config_dict.data_path + 'cifar-10-batches-py/test_batch')
        validation_array, validation_labels = read_images(config_dict, path_list, shuffle=False, is_random_label=False)
        # validation_array = whitening_image(validation_array, config_dict)
    elif config_dict.dataset == 'captcha':
        if not os.path.exists(config_dict.val_data_path):
            raise ValueError('images_path is not exist.')

        images = []
        labels = []
        images_path = os.path.join(config_dict.data_path, '*.jpg')
        count = 0
        for image_file in glob.glob(images_path):
            count += 1
            if count % 1000 == 0:
                print('Load {} images.'.format(count))
            image = cv2.imread(image_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Assume the name of each image is imagexxx_label.jpg
            label = int(image_file.split('_')[-1].split('.')[0])
            images.append(image)
            labels.append(label)
        validation_array = np.array(images)
        validation_labels = np.array(labels)
    elif config_dict.dataset == 'easy':
        if not os.path.exists(config_dict.val_data_path):
            raise ValueError('images_path is not exist.')

        images = []
        labels = []
        images_path = os.path.join(config_dict.data_path, '*.jpg')
        count = 0
        for image_file in glob.glob(images_path):
            count += 1
            if count % 100 == 0:
                print('Load {} images.'.format(count))
            image = cv2.imread(image_file)
            image = cv2.resize(image, (config_dict.input_resize_w, config_dict.input_resize_h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Assume the name of each image is imagexxx_label.jpg
            label = int(ord(image_file.split('_')[-2].split('/')[-1]) - 65)
            images.append(image)
            labels.append(label)
        validation_array = np.array(images)
        validation_labels = np.array(labels)

    return validation_array, validation_labels


def whitening_image(image_np, config_dict):
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        IMG_HEIGHT = config_dict.input_size_h
        IMG_WIDTH = config_dict.input_size_w
        IMG_DEPTH = config_dict.input_size_d
        std = np.max([np.std(image_np[i, ...]), 1.0 / np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i, ...] = (image_np[i, ...] - mean) / std
    return image_np


# only for cifar10
def read_images(config_dict, address_list, shuffle=True, is_random_label=False):
    data = np.array([]).reshape([0, config_dict.input_size_w * config_dict.input_size_h * config_dict.input_size_d])
    label = np.array([])
    for address in address_list:
        print('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch_cifar10(address, is_random_label)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))
    num_data = len(label)
    IMG_HEIGHT = config_dict.input_size_h
    IMG_WIDTH = config_dict.input_size_w
    IMG_DEPTH = config_dict.input_size_d
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if shuffle is True:
        print('Shuffling')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]
    data = data.astype(np.float32)
    return data, label


def _read_one_batch_cifar10(path, is_random_label):
    fo = open(path, 'rb')
    # python3在读文件时需要指定编码
    dicts = pickle.load(fo, encoding='iso-8859-1')
    fo.close()
    data = dicts['data']
    # for test, should not use!
    if is_random_label is False:
        label = np.array(dicts['labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label


'''
***************************************************************************************
'''


def _random_rotate(image, rotate_prob=0.5, rotate_angle_max=30,
                   interpolation='BILINEAR'):
    """Rotates the given image using the provided angle.

    Args:
        image: An image of shape [height, width, channels].
        rotate_prob: The probability to roate.
        rotate_angle_angle: The upper bound of angle to ratoted.
        interpolation: One of 'BILINEAR' or 'NEAREST'.(双线性插值和最邻近插值)

    Returns:
        The rotated image.
    """

    def _rotate():
        rotate_angle = tf.random_uniform([], minval=-rotate_angle_max,
                                         maxval=rotate_angle_max,
                                         dtype=tf.float32)
        rotate_angle = tf.div(tf.multiply(rotate_angle, math.pi), 180.)
        rotated_image = tf.contrib.image.rotate([image], [rotate_angle],
                                                interpolation=interpolation)
        return tf.squeeze(rotated_image)

    rand = tf.random_uniform([], minval=0, maxval=1)
    return tf.cond(tf.greater(rand, rotate_prob), lambda: image, _rotate)


def _border_expand(image, mode='CONSTANT', constant_values=255):
    """Expands the given image.

    Args:
        Args:
        image: A 3-D image `Tensor`.
        output_height: The height of the image after Expanding.
        output_width: The width of the image after Expanding.
        resize: A boolean indicating whether to resize the expanded image
            to [output_height, output_width, channels] or not.

    Returns:
        expanded_image: A 3-D tensor containing the resized image.
    """
    # todo: 这种闭包形式的用法
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    def _pad_left_right():
        pad_left = tf.floordiv(height - width, 2)
        pad_right = height - width - pad_left
        return [[0, 0], [pad_left, pad_right], [0, 0]]

    def _pad_top_bottom():
        pad_top = tf.floordiv(width - height, 2)
        pad_bottom = width - height - pad_top
        return [[pad_top, pad_bottom], [0, 0], [0, 0]]

    paddings = tf.cond(tf.greater(height, width),
                       _pad_left_right,
                       _pad_top_bottom)
    # expanding want to make w=h
    expanded_image = tf.pad(image, paddings, mode=mode,
                            constant_values=constant_values)
    return expanded_image


def _smallest_size_at_least(height, width, smallest_side):
    # 以给定的最短边并保持原图像横纵比计算新的w h
    """Computes new shape with the smallest side equal to `smallest_side`.

    Computes new shape with the smallest side equal to `smallest_side` while
    preserving the original aspect ratio.

    Args:
      height: an int32 scalar tensor indicating the current height.
      width: an int32 scalar tensor indicating the current width.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      new_height: an int32 scalar tensor indicating the new height.
      new_width: and int32 scalar tensor indicating the new width.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    # todo: learn to use lambda
    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(tf.rint(height * scale))
    new_width = tf.to_int32(tf.rint(width * scale))
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    # 保持原横纵比resize图像
    """Resize images preserving the original aspect ratio.

    Args:
      image: A 3-D image `Tensor`.
      smallest_side: A python integer or scalar `Tensor` indicating the size of
        the smallest side after resize.

    Returns:
      resized_image: A 3-D tensor containing the resized image.
    """
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                             align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def _fixed_sides_resize(image, output_height, output_width):
    """Resize images by fixed sides.

    Args:
        image: A 3-D image `Tensor`.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.

    Returns:
        resized_image: A 3-D tensor containing the resized image.
    """
    output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
    output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_nearest_neighbor(
        image, [output_height, output_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
      image: an image of shape [height, width, channels].
      offset_height: a scalar tensor indicating the height offset.
      offset_width: a scalar tensor indicating the width offset.
      crop_height: the height of the cropped image.
      crop_width: the width of the cropped image.

    Returns:
      the cropped (and resized) image.

    Raises:
      InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)


def _random_crop(image_list, crop_height, crop_width):
    """Crops the given list of images.

    The function applies the same crop to each image in the list. This can be
    effectively applied when there are multiple image inputs of the same
    dimension such as:

      image, depths, normals = _random_crop([image, depths, normals], 120, 150)

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the new height.
      crop_width: the new width.

    Returns:
      the image_list with cropped images.

    Raises:
      ValueError: if there are multiple image inputs provided with different size
        or the images are smaller than the crop dimensions.
    """
    if not image_list:
        raise ValueError('Empty image_list.')

    # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    # Create a random bounding box.
    #
    # Use tf.random_uniform and not numpy.random.rand as doing the former would
    # generate random numbers at graph eval time, unlike the latter which
    # generates random numbers at graph definition time.
    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
    with tf.control_dependencies(asserts):
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width,
                  crop_height, crop_width) for image in image_list]


# todo: where the mean and std from?
def _normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Normalizes an image."""
    image = tf.to_float(image)
    return tf.div(tf.div(image, 255.) - mean, std)


def _mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.

    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.

    Returns:
      the centered image.

    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def preprocess_for_train(image,
                         output_height,
                         output_width,
                         border_expand=False, normalize=True,
                         preserving_aspect_ratio_resize=False,
                         dataset_config=None):
    """Preprocesses the given image for training.

    Note that the actual resizing scale is sampled from
      [`resize_size_min`, `resize_size_max`].

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.

          The output_width and output_height should be smaller than resize_side_min!

      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing.

    Returns:
      A preprocessed image.
    """
    resize_side_min = dataset_config._RESIZE_SIDE_MIN
    resize_side_max = dataset_config._RESIZE_SIDE_MAX

    # todo: set rotate a switch
    # image = _random_rotate(image, rotate_angle_max=20)
    if border_expand:
        image = _border_expand(image)

    # 保留横纵比的resize
    if preserving_aspect_ratio_resize:
        # resize_side: resize后的最短边
        resize_side = tf.random_uniform(
            [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)

        image = _aspect_preserving_resize(image, resize_side)
    else:
        # todo: make it can set fixed resize
        image = _fixed_sides_resize(image, resize_side_min, resize_side_min)
    image = _random_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    # todo: set a switch
    image = tf.image.random_flip_left_right(image)
    if normalize:
        return _normalize(image)
    return _mean_image_subtraction(image, [dataset_config._R_MEAN, dataset_config._G_MEAN, dataset_config._B_MEAN])


def _central_crop(image_list, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
      image_list: a list of image tensors of the same dimension but possibly
        varying channel.
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      the list of cropped images.
    """
    outputs = []
    for image in image_list:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        offset_height = (image_height - crop_height) / 2
        offset_width = (image_width - crop_width) / 2

        outputs.append(_crop(image, offset_height, offset_width,
                             crop_height, crop_width))
    return outputs


def preprocess_for_eval(image, output_height, output_width, resize_side,
                        border_expand=False, normalize=True,
                        preserving_aspect_ratio_resize=False,
                        dataset_config=None):
    """Preprocesses the given image for evaluation.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      resize_side: The smallest side of the image for aspect-preserving resizing.

    Returns:
      A preprocessed image.
    """
    if border_expand:
        image = _border_expand(image)
    if preserving_aspect_ratio_resize:
        image = _aspect_preserving_resize(image, resize_side)
    else:
        image = _fixed_sides_resize(image, resize_side, resize_side)
    image = _central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    if normalize:
        return _normalize(image)
    return _mean_image_subtraction(image, [dataset_config._R_MEAN, dataset_config._G_MEAN, dataset_config._B_MEAN])


def preprocess_image(image, output_height, output_width, is_training=False,
                     border_expand=False, normalize=False,
                     preserving_aspect_ratio_resize=False,
                     dataset_config=None):
    """Preprocesses the given image.

    Args:
      image: A `Tensor` representing an image of arbitrary size.
      output_height: The height of the image after preprocessing.
      output_width: The width of the image after preprocessing.
      is_training: `True` if we're preprocessing the image for training and
        `False` otherwise.
      resize_side_min: The lower bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, then this value
        is used for rescaling.
      resize_side_max: The upper bound for the smallest side of the image for
        aspect-preserving resizing. If `is_training` is `False`, this value is
        ignored. Otherwise, the resize side is sampled from
          [resize_size_min, resize_size_max].

    Returns:
      A preprocessed image.
    """
    resize_side_min = dataset_config._RESIZE_SIDE_MIN
    resize_side_max = dataset_config._RESIZE_SIDE_MAX

    if is_training:
        return preprocess_for_train(image, output_height, output_width,
                                    border_expand, normalize,
                                    preserving_aspect_ratio_resize,
                                    dataset_config)
    else:
        return preprocess_for_eval(image, output_height, output_width,
                                   resize_side_min, border_expand, normalize,
                                   preserving_aspect_ratio_resize,
                                   dataset_config)


def preprocess_images(images, output_height, output_width,
                      is_training=False,
                      border_expand=False, normalize=True,
                      preserving_aspect_ratio_resize=False,
                      dataset_config=None):
    """Preprocesses the given image.

    Args:
        images: A `Tensor` representing a batch of images of arbitrary size.
        output_height: The height of the image after preprocessing.
        output_width: The width of the image after preprocessing.
        is_training: `True` if we're preprocessing the image for training and
            `False` otherwise.
        resize_side_min: The lower bound for the smallest side of the image
            for aspect-preserving resizing. If `is_training` is `False`, then
            this value is used for rescaling.
        resize_side_max: The upper bound for the smallest side of the image
            for aspect-preserving resizing. If `is_training` is `False`, this
            value is ignored. Otherwise, the resize side is sampled from
            [resize_size_min, resize_size_max].

    Returns:
        A  batch of preprocessed images.
    """
    # resize_side_min = dataset_config._RESIZE_SIDE_MIN
    # resize_side_max = dataset_config._RESIZE_SIDE_MAX

    images = tf.cast(images, tf.float32)

    def _preprocess_image(image):
        return preprocess_image(image, output_height, output_width,
                                is_training, border_expand, normalize,
                                preserving_aspect_ratio_resize,
                                dataset_config)

    return tf.map_fn(_preprocess_image, elems=images)


def border_expand(image, mode='CONSTANT', constant_values=255,
                  resize=False, output_height=None, output_width=None,
                  channels=3):
    """Expands (and resize) the given image."""
    expanded_image = _border_expand(image, mode, constant_values)
    if resize:
        if output_height is None or output_width is None:
            raise ValueError('`output_height` and `output_width` must be '
                             'specified in the resize case.')
        expanded_image = _fixed_sides_resize(expanded_image, output_height,
                                             output_width)
        expanded_image.set_shape([output_height, output_width, channels])
    return expanded_image
