# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.7.14
# state: modified
# 把最简单的分类captcha_images的simple_model从ckpt转pb

import tensorflow as tf
from tensorflow.python.framework import graph_util

with tf.Session() as sess:
    # Load .ckpt file
    saver = tf.train.import_meta_graph('../exp_output/captcha_simple_model/ckpt/model.ckpt.meta')
    saver.restore(sess, '../exp_output/captcha_simple_model/ckpt/model.ckpt')

    # Save as .pb file
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        ['classes']
    )
    with tf.gfile.GFile('../exp_output/captcha_simple_model/ckpt/convert_from_ckpt.pb', 'wb') as fid:
        serialized_graph = output_graph_def.SerializeToString()
        fid.write(serialized_graph)
