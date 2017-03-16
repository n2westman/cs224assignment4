import tensorflow as tf
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops


import logging


def maxout(inputs,
           num_units,
           axis=None,
           outputs_collections=None,
           scope=None):
  """Adds a maxout op which is a max pooling performed in filter/channel
  dimension. This can also be used after fully-connected layers to reduce
  number of features.
  Args:
    inputs: A Tensor on which maxout will be performed
    num_units: Specifies how many features will remain after max pooling at the
      channel dimension. This must be multiple of number of channels.
    axis: The dimension where max pooling will be performed. Default is the
      last dimension.
    outputs_collections: The collections to which the outputs are added.
    scope: Optional scope for name_scope.
  Returns:
    A `Tensor` representing the results of the pooling operation.
  Raises:
    ValueError: if num_units is not multiple of number of features.
    """
  with ops.name_scope(scope, 'MaxOut', [inputs]) as sc:
    inputs = ops.convert_to_tensor(inputs)
    shape = inputs.get_shape().as_list()
    if axis is None:
      # Assume that channel is the last dimension
      axis = -1
    num_channels = shape[axis]
    if num_channels % num_units:
      raise ValueError('number of features({}) is not '
                       'a multiple of num_units({})'
              .format(num_channels, num_units))
    shape[axis] = -1
    shape += [num_channels // num_units]
    outputs = math_ops.reduce_max(gen_array_ops.reshape(inputs, shape), -1,
                                  keep_dims=False)
    return utils.collect_named_outputs(outputs_collections, sc, outputs)

def highway_maxout(hidden_size, pool_size):
  """highway maxout network."""

  def compute(U, h, u_s, u_e):
    """
    Computes value of all u_t in U given current u_s and u_e.
    :args: U: coattention matrix

    :return: a score matrix of size [batch_size x max_context_words x l]
    """

    p = 4
    l = h.get_shape().as_list()[1]
    max_context_words = U.get_shape().as_list()[1]

    initializer = tf.contrib.layers.xavier_initializer()

    W_d = tf.get_variable('W_d', shape=(3*l, l), initializer=initializer, dtype=tf.float32)
    r = tf.tanh(tf.matmul(tf.concat([h, u_s, u_e], 1), W_d))

    u_r_concat = tf.reshape(tf.concat([U, tf.tile(tf.expand_dims(r, 1), [1, max_context_words, 1])], 2), [-1, 2*l])

    W_1 = tf.get_variable('W_1', shape=(2*l, l*p), initializer=initializer, dtype=tf.float32)
    b_1 = tf.get_variable('b_1', shape=(l*p), initializer=initializer, dtype=tf.float32)

    m_t1 = tf.reshape(tf.matmul(u_r_concat, W_1) + b_1, [-1, max_context_words, l, p])
    m_t1 = tf.reduce_max(m_t1, axis=3)

    W_2 = tf.get_variable('W_2', shape=(l, l*p), initializer=initializer, dtype=tf.float32)
    b_2 = tf.get_variable('b_2', shape=(l*p), initializer=initializer, dtype=tf.float32)

    m_t2 = tf.reshape(tf.matmul(tf.reshape(m_t1, [-1, l]), W_2) + b_2, [-1, max_context_words, l, p])
    m_t2 = tf.reduce_max(m_t2, axis=3)

    m = tf.concat([m_t1, m_t2], 2)

    W_3 = tf.get_variable('W_3', shape=(2*l, p), initializer=initializer, dtype=tf.float32)
    b_3 = tf.get_variable('b_3', shape=(p), initializer=initializer, dtype=tf.float32)

    scores = tf.reshape(tf.matmul(tf.reshape(m, [-1, 2*l]), W_3) + b_3, [-1, max_context_words, 1, p])
    scores = tf.reduce_max(scores, axis=3)

    return scores

  return compute
