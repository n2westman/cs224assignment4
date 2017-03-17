try:
    import google3
except:
    pass

import tensorflow as tf

def highway_maxout(hidden_size, pool_size):
  """highway maxout network."""

  def compute(U, h, u_s, u_e):
    """
    Computes value of all u_t in U given current u_s and u_e.
    :args: U: coattention matrix

    :return: a score matrix of size [batch_size x max_context_words x l]
    """

    p = pool_size
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
