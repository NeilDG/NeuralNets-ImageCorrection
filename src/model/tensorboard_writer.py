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
      
def logGradients(loss, layerName, labelName):
    gr = tf.get_default_graph()
    weight = gr.get_tensor_by_name(layerName)
    grad = tf.gradients(loss, weight)[0]
    mean = tf.reduce_mean(tf.abs(grad))
    tf.summary.scalar('mean_weight_{}'.format(labelName), mean)
    tf.summary.histogram('hist_weights_{}'.format(labelName), grad)