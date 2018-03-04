import tensorflow as tf
from tensorflow.contrib.layers import flatten

BETA=0.00001
MU = 0
SIGMA = 0.1

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def feedforward_cnn01(x, keep_prob, num_channels):
   # Hyperparameters

    with tf.name_scope('conv1'):
        conv1_W = tf.Variable(tf.truncated_normal(shape=(7, 7, num_channels, 16), mean = MU, stddev = SIGMA))
        conv1_b = tf.Variable(tf.zeros(16))
        conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 2, 2, 1], padding='SAME') + conv1_b
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
        variable_summaries(conv1_W, 'weights')
        variable_summaries(conv1_b, 'bias')

    with tf.name_scope('conv2'):
        conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 16, 16), mean = MU, stddev = SIGMA))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='SAME') + conv2_b
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


    with tf.name_scope('conv3'):
        conv3_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 16, 32), mean = MU, stddev = SIGMA))
        conv3_b = tf.Variable(tf.zeros(32))
        conv3   = tf.nn.conv2d(conv2, conv3_W, strides=[1, 1, 1, 1], padding='SAME') + conv3_b
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    with tf.name_scope('conv4'):
        conv4_W = tf.Variable(tf.truncated_normal(shape=(3, 3, 32, 64), mean = MU, stddev = SIGMA))
        conv4_b = tf.Variable(tf.zeros(64))
        conv4   = tf.nn.conv2d(conv3, conv4_W, strides=[1, 1, 1, 1], padding='SAME') + conv4_b
        conv4 = tf.nn.relu(conv4)
        conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

        flat  = flatten(conv4)

    with tf.name_scope('fc1'):
        fc1_W = tf.Variable(tf.truncated_normal(shape=(64 * 3 * 6, 64), mean = MU, stddev = SIGMA))
        fc1_b = tf.Variable(tf.zeros(64))
        fc1   = tf.matmul(flat, fc1_W) + fc1_b
        fc1    = tf.nn.relu(fc1)
        fc1 = tf.nn.dropout(fc1, keep_prob)

    with tf.name_scope('fc2'):
        fc2_W  = tf.Variable(tf.truncated_normal(shape=(64, 32), mean = MU, stddev = SIGMA))
        fc2_b  = tf.Variable(tf.zeros(32))
        fc2    = tf.matmul(fc1, fc2_W) + fc2_b
        fc2    = tf.nn.relu(fc2)
        fc2 = tf.nn.dropout(fc2, keep_prob)

        fc3_W  = tf.Variable(tf.truncated_normal(shape=(32, 1), mean = MU, stddev = SIGMA))
        fc3_b  = tf.Variable(tf.zeros(1))

    with tf.name_scope('out'):
        out = tf.squeeze(tf.tanh(tf.matmul(fc2, fc3_W) + fc3_b), 1)

    l2reg = BETA * tf.nn.l2_loss(conv1_W) + BETA * tf.nn.l2_loss(conv1_b)
    l2reg = l2reg + BETA * tf.nn.l2_loss(conv2_W) + BETA * tf.nn.l2_loss(conv2_b)
    l2reg = l2reg + BETA * tf.nn.l2_loss(fc1_W) + BETA * tf.nn.l2_loss(fc1_b)
    l2reg = l2reg + BETA * tf.nn.l2_loss(fc2_W) + BETA * tf.nn.l2_loss(fc2_b)
    l2reg = l2reg + BETA * tf.nn.l2_loss(fc3_W) + BETA * tf.nn.l2_loss(fc3_b)

    return out, l2reg
