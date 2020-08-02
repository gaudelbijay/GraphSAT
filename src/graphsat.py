import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.initializers import glorot_uniform, Zeros
from tensorflow.keras.layers import Input, Dense, Dropout, Layer
from tensorflow.keras.regularizers import l2


def GraphSAT(feature_dim,
             neighbor_num,
             n_att_head,
             att_embedding_size,
             n_classes,
             use_bias=True,
             activation=tf.nn.relu,
             aggregator_type='mean',
             dropout_rate=0.0,
             l2_reg=0):

    features = Input(shape=(feature_dim,))
    node_input = Input(shape=(1,), dtype=tf.int64)
    neighbor_input = [Input(shape=(l,), dtype=tf.int64) for l in neighbor_num]

    if aggregator_type == 'mean':
        aggregator = MeanAggregator
    else:
        aggregator = PoolingAggregator

    h = features
    for i in range(0, len(neighbor_num)):
        if i == len(neighbor_num) - 1:
            activation = tf.nn.softmax
            att_embedding_size = n_classes
        h = aggregator(att_embedding_size=att_embedding_size, head_num=n_att_head,
                       activation=activation, l2_reg=l2_reg,
                       use_bias=use_bias, dropout_rate=dropout_rate,
                       aggregator=aggregator_type)([h, node_input,
                                                    neighbor_input[i]])  #

    output = h
    input_list = [features, node_input] + neighbor_input
    model = Model(input_list, outputs=output)
    return model


class PoolingAggregator(Layer):

    def __init__(self, att_embedding_size=8, head_num=8, activation=tf.nn.relu,
                 l2_reg=0.0, use_bias=False, dropout_rate=0.5,
                 seed=1024, aggregator='mean', **kwargs):
        if head_num <= 0:
            raise ValueError('head_num must be a int >0')

        self.att_embedding_size = att_embedding_size
        self.head_num = head_num
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.l2_reg = l2_reg
        self.use_bias = use_bias
        self.pooling = aggregator
        self.seed = seed
        super(PoolingAggregator, self).__init__(**kwargs)

    def build(self, input_shape):
        X, N, neigh = input_shape
        embedding_size = X[-1]

        # For Neighbor pooling
        self.dense_layer = [Dense(embedding_size, activation=tf.nn.relu, use_bias=True,
                                  kernel_regularizer=l2(self.l2_reg))]

        # Transforming weight  to learn state of features.
        self.weight = self.add_weight(name='weight',
                                      shape=[embedding_size, self.att_embedding_size * self.head_num],
                                      dtype=tf.float32, regularizer=l2(self.l2_reg),
                                      initializer=tf.keras.initializers.glorot_uniform()
                                      )

        # node attention weight
        self.att_self_weight = self.add_weight(name='att_self_weight',
                                               shape=[1, self.head_num, self.att_embedding_size],
                                               dtype=tf.float32, regularizer=l2(self.l2_reg),
                                               initializer=tf.keras.initializers.glorot_uniform()
                                               )
        # neighbor attention weight
        self.att_neigh_weight = self.add_weight(name='att_self_weight',
                                                shape=[1, self.head_num, self.att_embedding_size],
                                                dtype=tf.float32, regularizer=l2(self.l2_reg),
                                                initializer=tf.keras.initializers.glorot_uniform()
                                                )
        if self.use_bias:
            self.bias_weight = self.add_weight(name='bias',
                                               shape=[1, self.head_num, self.att_embedding_size],
                                               dtype=tf.float32, regularizer=Zeros()
                                               )

        self.self_dropout = Dropout(self.dropout_rate)
        self.neigh_dropout = Dropout(self.dropout_rate)
        self.att_dropout = Dropout(self.dropout_rate)

    def call(self, inputs):
        features, node, neighbor = inputs
        # print(features, node, inputs)
        embedding_size = features.shape[-1]

        node_feat = tf.nn.embedding_lookup(features, node)
        neigh_feat = tf.nn.embedding_lookup(features, neighbor)

        node_feat = tf.squeeze(node_feat, axis=1)
        dims = tf.shape(neigh_feat)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(neigh_feat, (batch_size * num_neighbors, embedding_size))

        for l in self.dense_layer:
            h_reshaped = l(h_reshaped)
        neigh_feat = tf.reshape(h_reshaped, (batch_size, num_neighbors, h_reshaped.shape[-1]))
        if self.pooling == 'mean':
            neigh_feat = tf.reduce_mean(neigh_feat, axis=1)
        else:
            neigh_feat = tf.reduce_max(neigh_feat, axis=1)

        node_feat = tf.matmul(node_feat, self.weight)
        node_feat = tf.reshape(node_feat, [-1, self.head_num, self.att_embedding_size])
        # print('node feat:   ', node_feat)
        neigh_feat = tf.matmul(neigh_feat, self.weight)
        neigh_feat = tf.reshape(neigh_feat, [-1, self.head_num, self.att_embedding_size])
        att_for_self = tf.reduce_sum(node_feat * self.att_self_weight, axis=-1, keepdims=True)
        att_for_neigh = tf.reduce_sum(neigh_feat * self.att_neigh_weight, axis=-1, keepdims=True)

        dense = tf.transpose(att_for_self, [1, 0, 2]) + tf.transpose(att_for_neigh, [1, 2, 0])
        dense = tf.nn.leaky_relu(dense, alpha=0.2)

        self.normalized_att_scores = tf.nn.softmax(dense, axis=-1, )
        self.normalized_att_scores = self.att_dropout(self.normalized_att_scores)

        node_feat = self.self_dropout(node_feat)

        result = tf.matmul(self.normalized_att_scores, tf.transpose(node_feat, [1, 0, 2]))
        result = tf.transpose(result, [1, 0, 2])

        if self.use_bias:
            result += self.bias_weight

        result = tf.reduce_mean(result, axis=1)

        if self.activation:
            result = self.activation(result)

        # result._use_learning_phase = True
        return result


class MeanAggregator(Layer):
    pass

