# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.14
# state: modified
# 用于测试模型

import os
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from datasets.create_classification_data import *


if_pb_model = False
flags = tf.app.flags
flags.DEFINE_string('model_ckpt_path', 'exp_output/captcha_simple_model/ckpt/model.ckpt', 'Path to model checkpoint.')
FLAGS = flags.FLAGS


def test_image():
    if if_pb_model:
        direct_pb_path = './exp_output/captcha_simple_model/pb_saved_direct/model.pb'
        # direct_pb_path = './exp_output/captcha_simple_model/ckpt/convert_from_ckpt.pb'
        if not os.path.exists(direct_pb_path):
            raise ValueError("'path_to_model.pb' is not exist.")

        model_graph = tf.Graph()
        with model_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(direct_pb_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                print(od_graph_def)
                tf.import_graph_def(od_graph_def, name='')

        # summaryWriter = tf.summary.FileWriter('exp_output/captcha_simple_model/', model_graph)

        inputs = model_graph.get_tensor_by_name('inputs:0')
        classes = model_graph.get_tensor_by_name('classes:0')

        with model_graph.as_default():
            with tf.Session(graph=model_graph) as sess:
                for i in range(10):
                    label = np.random.randint(0, 10)
                    image = generate_captcha(str(label))
                    image_np = np.expand_dims(image, axis=0)
                    predicted_label = sess.run(classes,
                                               feed_dict={inputs: image_np})
                    print(predicted_label, ' vs ', label)

    with tf.Session() as sess:

        ckpt_path = FLAGS.model_ckpt_path

        # reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
        # var_to_shape_map = reader.get_variable_to_shape_map()
        # for key in var_to_shape_map:
        #     print("tensor_name: ", key)

        saver = tf.train.import_meta_graph(ckpt_path + '.meta')
        saver.restore(sess, ckpt_path)

        inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
        classes = tf.get_default_graph().get_tensor_by_name('classes:0')

        for i in range(10):
            label = np.random.randint(0, 10)
            image = generate_captcha(str(label))
            image_np = np.expand_dims(image, axis=0)
            predicted_label = sess.run(classes,
                                       feed_dict={inputs: image_np})
            print(predicted_label, ' vs ', label)


if __name__ == '__main__':
    test_image()
