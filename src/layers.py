from inits import *
import tensorflow as tf
from abc import abstractmethod

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform([noise_shape], dtype=tf.float64)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    res = tf.sparse_retain(x, dropout_mask)
    res /= keep_prob
    return res


def dot(x, y, sparse):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    def __init__(self, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = []

    @abstractmethod
    def _call(self, inputs):
        pass

    def __call__(self, inputs):
        outputs = self._call(inputs)
        return outputs


class GCNLayer(Layer):
    def __init__(self, input_dim, output_dim, adj, dropout, sparse=False, feature_nnz=0, act=tf.nn.relu, name=None):
        super(GCNLayer, self).__init__(name)
        self.adj = adj
        self.dropout = dropout
        self.sparse = sparse
        self.feature_nnz = feature_nnz
        self.act = act
        with tf.variable_scope(self.name):
            self.weights = glorot([input_dim, output_dim], name='weight')
            self.vars = [self.weights]

    def _call(self, inputs):
        x = inputs
        x = sparse_dropout(x, 1 - self.dropout, self.feature_nnz) if self.sparse else tf.nn.dropout(x, 1 - self.dropout)
        x = dot(x, self.weights, sparse=self.sparse)
        x = dot(self.adj, x, sparse=True)
        return self.act(x)


class LPALayer(Layer):
    def __init__(self, adj, name=None):
        super(LPALayer, self).__init__(name)
        self.adj = adj

    def _call(self, inputs):
        output = dot(self.adj, inputs, sparse=True)
        return output
