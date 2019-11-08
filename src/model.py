from layers import *
from inits import *


class GCN_LPA(object):
    def __init__(self, args, features, labels, adj):
        self.args = args
        self.vars = []  # for computing l2 loss

        self._build_inputs(features, labels)
        self._build_edges(adj)
        self._build_gcn(features[2][1], labels.shape[1], features[0].shape[0])
        self._build_lpa()
        self._build_train()
        self._build_eval()

    def _build_inputs(self, features, labels):
        self.features = tf.SparseTensor(*features)
        self.labels = tf.constant(labels, dtype=tf.float64)
        self.label_mask = tf.placeholder(tf.float64, shape=labels.shape[0])
        self.dropout = tf.placeholder(tf.float64)

    def _build_edges(self, adj):
        edge_weights = glorot(shape=[adj[0].shape[0]])
        self.adj = tf.SparseTensor(adj[0], edge_weights, adj[2])
        self.normalized_adj = tf.sparse_softmax(self.adj)
        self.vars.append(edge_weights)

    def _build_gcn(self, feature_dim, label_dim, feature_nnz):
        hidden_list = []

        if self.args.gcn_layer == 1:
            gcn_layer = GCNLayer(input_dim=feature_dim, output_dim=label_dim, adj=self.normalized_adj,
                                 dropout=self.dropout, sparse=True, feature_nnz=feature_nnz, act=lambda x: x)
            self.outputs = gcn_layer(self.features)
            self.vars.extend(gcn_layer.vars)
        else:
            gcn_layer = GCNLayer(input_dim=feature_dim, output_dim=self.args.dim, adj=self.normalized_adj,
                                 dropout=self.dropout, sparse=True, feature_nnz=feature_nnz)
            hidden = gcn_layer(self.features)
            hidden_list.append(hidden)
            self.vars.extend(gcn_layer.vars)

            for _ in range(self.args.gcn_layer - 2):
                gcn_layer = GCNLayer(input_dim=self.args.dim, output_dim=self.args.dim, adj=self.normalized_adj,
                                     dropout=self.dropout)
                hidden = gcn_layer(hidden_list[-1])
                hidden_list.append(hidden)
                self.vars.extend(gcn_layer.vars)

            gcn_layer = GCNLayer(input_dim=self.args.dim, output_dim=label_dim, adj=self.normalized_adj,
                                 dropout=self.dropout, act=lambda x: x)
            self.outputs = gcn_layer(hidden_list[-1])
            self.vars.extend(gcn_layer.vars)

        self.prediction = tf.nn.softmax(self.outputs, axis=-1)

    def _build_lpa(self):
        label_mask = tf.expand_dims(self.label_mask, -1)
        input_labels = label_mask * self.labels
        label_list = [input_labels]

        for _ in range(self.args.lpa_iter):
            lp_layer = LPALayer(adj=self.normalized_adj)
            hidden = lp_layer(label_list[-1])
            label_list.append(hidden)
        self.predicted_label = label_list[-1]

    def _build_train(self):
        # GCN loss
        self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=self.labels)
        self.loss = tf.reduce_sum(self.loss * self.label_mask) / tf.reduce_sum(self.label_mask)

        # LPA loss
        lpa_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.predicted_label, labels=self.labels)
        lpa_loss = tf.reduce_sum(lpa_loss * self.label_mask) / tf.reduce_sum(self.label_mask)
        self.loss += self.args.lpa_weight * lpa_loss

        # L2 loss
        for var in self.vars:
            self.loss += self.args.l2_weight * tf.nn.l2_loss(var)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.lr).minimize(self.loss)

    def _build_eval(self):
        correct_prediction = tf.equal(tf.argmax(self.outputs, 1), tf.argmax(self.labels, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float64)
        self.accuracy = tf.reduce_sum(correct_prediction * self.label_mask) / tf.reduce_sum(self.label_mask)
