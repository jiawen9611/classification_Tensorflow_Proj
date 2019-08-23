# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.14
# state: modified
# 用于测试模型

import os
# import numpy as np
import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
from datasets.create_classification_data import *
import yaml
import time
from easydict import EasyDict
from datasets.data_preprocess import *

# !!!!!!!!!only for easy_resnet50
if_pb_model = False
flags = tf.app.flags
flags.DEFINE_string('model_ckpt_path', 'exp_output/easy_resnet50/ckpt/model.ckpt', 'Path to model checkpoint.')
flags.DEFINE_string('test_img_path', 'datasets/easy/test_with_label/', 'dataset.')
flags.DEFINE_string('config_path', 'exp_configs/easy_resnet50/config.yaml', 'config_path.')
FLAGS = flags.FLAGS


def test_model():
    ckpt_path = FLAGS.model_ckpt_path
    config_path = FLAGS.config_path
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = EasyDict(config)
    saver = tf.train.import_meta_graph(ckpt_path + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, ckpt_path)

        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
        classes = tf.get_default_graph().get_tensor_by_name('classes:0')
        is_training = tf.get_default_graph().get_tensor_by_name('is_training:0')

        start_time = time.time()
        images_path = os.path.join(FLAGS.test_img_path, '*.jpg')
        for image_file in glob.glob(images_path):
            image = cv2.imread(image_file)
            image = cv2.resize(image, (config.input_resize_w, config.input_resize_h))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.expand_dims(image, axis=0)
            predicted_label = sess.run(classes, feed_dict={inputs: image, is_training: False})
            print(predicted_label, ' vs ', image_file)
        time_count = time.time() - start_time
        examples_per_sec = config.val_num / time_count
        print("speed:", examples_per_sec)


if __name__ == '__main__':
    test_model()
