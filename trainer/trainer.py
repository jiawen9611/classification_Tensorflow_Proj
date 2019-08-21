# -*-coding:utf-8-*-

# coder: Jiawen Zhu
# date: 2019.6.9
# state: half modified


# import torch.backends.cudnn as cudnn
# import time
import models
# from models import *
from running_logger.create_logger import *
from datasets.data_preprocess import *
# from datasets.record_dataset import *
# from tensorboardX import SummaryWriter
from tensorflow.python.framework import graph_util


class Trainer():
    def __init__(self, output_path=None, config=None):
        self.output_path = output_path
        self.config = config
        self.logger = Logger(log_file_name=output_path + '/log.txt',
                             log_level=logging.DEBUG, logger_name="").get_log()
        self.image_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, config.input_resize_h,
                                                                         config.input_resize_w,
                                                                         config.input_size_d], name='inputs')
        self.label_placeholder = tf.placeholder(dtype=tf.int32, shape=[None])
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_device
        if config.architecture == 'simple_model':
            self.cls_model = models.simple_model.Model(is_training=self.is_training, num_classes=config.num_classes)
        elif config.architecture == 'resnet_v1_50':
            if config.dataset == 'cifar10' or 'captcha':
                self.cls_model = models.resnet_v1_50.Model(is_training=self.is_training, num_classes=config.num_classes,
                                                           fixed_resize_side=32,
                                                           default_image_size=32,
                                                           dataset_config=self.config)
            else:
                self.cls_model = models.resnet_v1_50.Model(is_training=True, num_classes=self.config.num_classes,
                                                           dataset_config=self.config)

    def start_train_and_val(self):
        if self.config.architecture == 'resnet_v1_50':
            preprocessed_inputs = self.image_placeholder
        else:
            preprocessed_inputs = self.cls_model.preprocess(self.image_placeholder)
        prediction_dict = self.cls_model.predict(preprocessed_inputs)
        regu_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        loss_dict = self.cls_model.loss(prediction_dict, self.label_placeholder)
        loss_ = loss_dict['loss']
        loss = tf.add_n([loss_] + regu_losses)
        postprocessed_dict = self.cls_model.postprocess(prediction_dict)
        classes = postprocessed_dict['classes']
        classes = tf.cast(classes, tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(classes, self.label_placeholder), 'float'))
        global_step = tf.Variable(0, trainable=False)
        # learning_rate
        learning_rate = tf.train.exponential_decay(self.config.lr_scheduler.init_lr, global_step, 2500, 0.9)
        # optimizer
        optimizer = tf.train.MomentumOptimizer(learning_rate, self.config.optimize.momentum)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
        # 如有BN，算子切勿使用次方式更新
        # train_op = optimizer.minimize(loss, global_step=global_step)

        saver = tf.train.Saver(max_to_keep=3)
        train_data, train_labels = read_train_data(self.config)
        val_data, val_labels = read_validation_data(self.config)

        # todo:区别
        # init = tf.global_variables_initializer()
        # init = tf.initialize_all_variables()
        with tf.Session() as sess:
            # sess.run(init)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            # todo
            # coord = tf.train.Coordinator()
            # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for i in range(self.config.train_steps):
                # train
                batch_images, batch_labels = generate_augment_train_batch(train_data, train_labels, self.config)
                train_dict = {self.image_placeholder: batch_images, self.label_placeholder: batch_labels,
                              self.is_training: True}
                _, loss_, acc_, lr_ = sess.run([train_op, loss, acc, learning_rate], feed_dict=train_dict)
                if i % 100 == 0:
                    train_text = 'step: {}, lr: {:.5f}, loss: {:.5f}, acc: {}'.format(
                        i + 1, lr_, loss_, acc_)
                    # print(train_text)
                    self.logger.info(train_text)
                # val
                if i > 100 and i % 1000 == 0:
                    num_batches = self.config.val_num // self.config.val_batch
                    order = np.random.choice(self.config.val_num, num_batches * self.config.val_batch)
                    vali_data_subset = val_data[order, ...]
                    vali_labels_subset = val_labels[order]
                    loss_list = []
                    acc_list = []
                    self.logger.info('×*×*×*×*×*×*×*×*×*×*×*Start test×*×*×*×*×*×*×*×*×*×*×*')
                    for step in range(num_batches):
                        offset = step * self.config.val_batch
                        val_feed_dict = {self.image_placeholder: vali_data_subset[
                                                                 offset:offset + self.config.val_batch, ...],
                                         self.label_placeholder: vali_labels_subset[
                                                                 offset:offset + self.config.val_batch],
                                         self.is_training: False
                                         }
                        val_loss, val_acc = sess.run([loss, acc], feed_dict=val_feed_dict)
                        loss_list.append(val_loss)
                        acc_list.append(val_acc)

                    val_text = 'val_loss: {:.5f}, val_acc: {}'.format(np.mean(loss_list), np.mean(acc_list))
                    self.logger.info(val_text)
            if not os.path.exists(self.config.ckpt_path):
                os.mkdir(self.config.ckpt_path)
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
            # coord.request_stop()
            # coord.join(threads)
