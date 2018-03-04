from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf
import numpy as np
from data import BatchGenerator
from model import feedforward_cnn01

flags = tf.flags

flags.DEFINE_integer("max_steps", 50,'Number of steps to run trainer.')
flags.DEFINE_string("train_data", '', '')
flags.DEFINE_string("valid_data", '', '')
flags.DEFINE_integer("batch_size", 64, '')
flags.DEFINE_integer("input_height", 180, '')
flags.DEFINE_integer("input_width", 320, '')
flags.DEFINE_integer("input_channels", 1, '')

flags.DEFINE_integer("test_every", 100, 'Number of steps to run trainer.')
flags.DEFINE_integer("display_every", 100, 'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.001,'Initial learning rate')
flags.DEFINE_string("summaries_dir", 'summaries','Summaries directory')

FLAGS = flags.FLAGS


def main(args):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    tf.logging.set_verbosity(tf.logging.INFO)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Input placeholders
        with tf.name_scope('input'):
            image_ = tf.placeholder(tf.float32,
                                   [None, FLAGS.input_height, FLAGS.input_width, FLAGS.input_channels],
                                   name='image')
            label_ = tf.placeholder(tf.float32, [None,], name='label')
            keep_prob_ = tf.placeholder(tf.float32)

        with tf.variable_scope('model'):
            prediction, l2reg = feedforward_cnn01(image_, keep_prob_, FLAGS.input_channels)
            mse_loss = tf.losses.mean_squared_error(label_, prediction)
            rmse = tf.metrics.root_mean_squared_error(label_, prediction)
            loss_op = tf.reduce_mean(mse_loss + l2reg)
            optimizer = tf.train.AdamOptimizer(learning_rate = FLAGS.learning_rate)
            train_op = optimizer.minimize(loss_op)

        tf.summary.scalar('Error', rmse[1])
        merged_summaries = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        valid_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/valid')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        train_data = BatchGenerator(FLAGS.train_data, FLAGS.batch_size, shuffle=True, infinite=True)
        valid_data = BatchGenerator(FLAGS.valid_data, FLAGS.batch_size, shuffle=False, infinite=False)

        for i in range(FLAGS.max_steps):
            image, label = next(train_data)
            _, rmse_error, summaries = sess.run([train_op, rmse, merged_summaries],
                                     feed_dict={image_: image, label_: label, keep_prob_: 0.5})
            train_writer.add_summary(summaries, i)

            if i % FLAGS.display_every == 0:
                tf.logging.info("Iter %d, Train error: %f", i, rmse_error[1]/float(FLAGS.batch_size))

            if i % FLAGS.test_every == 0:  # test-set accuracy
                tf.logging.info("Iter %d, Validation phase", i)
                total_err = 0.0
                num_samples = 0
                valid_data.reset()
                for image, label in valid_data:
                    err_value, summaries = sess.run([rmse, merged_summaries],
                                                    feed_dict={image_: image, label_: label, keep_prob_: 1.0})
                    valid_writer.add_summary(summaries, i)
                    total_err += err_value[1]
                    num_samples += image.shape[0]

                tf.logging.info("Validation error: %f", total_err/float(num_samples))

        train_writer.close()
        valid_writer.close()


if __name__ == '__main__':
    tf.app.run()
