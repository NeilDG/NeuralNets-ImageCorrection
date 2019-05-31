# -*- coding: utf-8 -*-
import tensorflow as tf

def variable_summaries(var, tag):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar('mean_{}'.format(tag), mean)
      
      with tf.name_scope('stddev_compute'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          
      tf.summary.scalar('stddev_{}'.format(tag), stddev)

      tf.summary.histogram('histogram_weights_{}'.format(tag), var)
      
def log_weights(train_vars):
    for i in train_vars:
        name = i.name.split(":")[0]
        value = i.value()
        tf.summary.histogram(name, value)