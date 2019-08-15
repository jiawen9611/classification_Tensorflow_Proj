# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.6.9
# state: half modified


import torch.backends.cudnn as cudnn
import time
import models
from models import *
from running_logger.create_logger import *
from datasets.data_preprocess import *
from datasets.record_dataset import *
from tensorboardX import SummaryWriter
from tensorflow.python.framework import graph_util


class Trainer():
    def __init__(self, output_path=None, config=None):
        self.output_path = output_path
        self.config = config
        self.logger = Logger(log_file_name=output_path + '/log.txt',
                             log_level=logging.DEBUG, logger_name="").get_log()
        self.train_image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, config.input_resize_h,
                                                                               config.input_resize_w,
                                                                               config.input_size_d], name='inputs')

        self.train_label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])

        # todo: shape = [None, input_size_h, input_size_w, input_size_d]
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device
        if config.architecture == 'simple_model':
            self.cls_model = models.simple_model.Model(is_training=True, num_classes=config.num_classes)
        if config.architecture == 'resnet_v1_50':
            self.cls_model = models.resnet_v1_50.Model(is_training=True, num_classes=config.num_classes,
                                                       dataset_config=config)

    def start_train_and_val(self):
        preprocessed_inputs = self.cls_model.preprocess(self.train_image_placeholder)
        prediction_dict = self.cls_model.predict(preprocessed_inputs)

        loss_dict = self.cls_model.loss(prediction_dict, self.train_label_placeholder)
        loss = loss_dict['loss']
        postprocessed_dict = self.cls_model.postprocess(prediction_dict)
        classes = postprocessed_dict['classes']
        classes= tf.cast(classes, tf.int32)
        classes_ = tf.identity(classes, name='classes')

        acc = tf.reduce_mean(tf.cast(tf.equal(classes, self.train_label_placeholder), 'float'))
        global_step = tf.Variable(0, trainable=False)
        validation_step = tf.Variable(0, trainable=False)
        # learning_rate
        learning_rate = tf.train.exponential_decay(self.config.lr_scheduler.init_lr, global_step, 150, 0.9)
        # optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, self.config.optimize.momentum)
        train_op = optimizer.minimize(loss, global_step)
        # val
        saver = tf.train.Saver(max_to_keep=3)
        train_data, train_labels = read_train_data(self.config)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(self.config.train_steps):
                batch_images, batch_labels = generate_augment_train_batch(train_data, train_labels, self.config)
                train_dict = {self.train_image_placeholder: batch_images, self.train_label_placeholder: batch_labels}
                _, loss_, acc_ = sess.run([train_op, loss, acc], feed_dict=train_dict)
                if i % 100 == 0:
                    train_text = 'step: {}, loss: {}, acc: {}'.format(
                        i + 1, loss_, acc_)
                    print(train_text)
            saver.save(sess, self.config.ckpt_path + 'model.ckpt')
            if self.config.save_pb_direct:
                graph_def = tf.get_default_graph().as_graph_def()
                output_graph_def = graph_util.convert_variables_to_constants(
                    sess,
                    graph_def,
                    ['classes']
                )
                with tf.gfile.GFile(self.config.pb_direct_path + 'model.pb', 'wb') as fid:
                    serialized_graph = output_graph_def.SerializeToString()
                    fid.write(serialized_graph)
