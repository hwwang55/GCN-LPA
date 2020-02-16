import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float64)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    init_range = np.sqrt(6.0 / np.sum(shape))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float64)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float64)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    initial = tf.ones(shape, dtype=tf.float64)
    return tf.Variable(initial, name=name)
