import numpy as np
import os
import tensorflow as tf
from neighbor_sampler import sample_neighs
import networkx as nx
from graphsat import GraphSAT
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model
import scipy.sparse as sp
from utils import preprocess_adj, plot_embeddings, load_data_v1, load_data

tf.compat.v1.disable_eager_execution()

if __name__ == "__main__":
    print('Tensorflow ', tf.__version__, ' is running: ')
    A, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_v1('cora')
    features /= features.sum(axis=1, ).reshape(-1, 1)
    G = nx.from_scipy_sparse_matrix(A, create_using=nx.DiGraph())
    # A = preprocess_adj(A)
    A = A + sp.eye(A.shape[0])
    indexes = np.arange(A.shape[0])
    neigh_number = [10, 10]
    neigh_maxlen = []
    features = np.squeeze(np.asarray(features))
    # print(features.shape)
    model_input = [A.toarray(), features, np.asarray(indexes, dtype=np.int64)]
    for num in neigh_number:
        sample_neigh, sample_neigh_len = sample_neighs(G, indexes, num, self_loop=False)
        model_input.extend([sample_neigh])
        neigh_maxlen.append(max(sample_neigh_len))

    model = GraphSAT(adj_dim=A.shape[0],
                     feature_dim=features.shape[-1],
                     neighbor_num=neigh_maxlen,
                     n_att_head=6,
                     att_embedding_size=256,
                     n_classes=y_train.shape[1],
                     use_bias=True,
                     activation=tf.nn.relu,
                     aggregator_type='mean',
                     dropout_rate=0.0,
                     l2_reg=2.5e-20, )
    model.compile(Adam(0.005), 'categorical_crossentropy',
                  weighted_metrics=['categorical_crossentropy', 'acc'])
    val_data = [model_input, y_val, val_mask]

    mc_callback = ModelCheckpoint('./best_model.h5',
                                  monitor='val_acc',
                                  save_best_only=True,
                                  save_weights_only=True)
    es_callback = EarlyStopping(monitor='val_acc', patience=50)
    print('start training')
    model.fit(model_input, y_train, sample_weight=train_mask, validation_data=val_data, batch_size=A.shape[0],
              epochs=100, shuffle=False, verbose=2, callbacks=[mc_callback, es_callback])
    model.load_weights('./best_model.h5')

    eval_results = model.evaluate(
        model_input, y_test, sample_weight=test_mask, batch_size=A.shape[0])
    print('Done.\n'
          'Test loss: {}\n'
          'Test weighted_loss: {}\n'
          'Test accuracy: {}'.format(*eval_results))

    gcn_embedding = model.layers[-1]
    embedding_model = Model(model.input, outputs=Lambda(lambda x: gcn_embedding.output)(model.input))
    embedding_weights = embedding_model.predict(model_input, batch_size=A.shape[0])
    print(embedding_weights.shape)
    y = np.genfromtxt("{}{}.content".format('../data/cora/', 'cora'), dtype=np.dtype(str))[:, -1]
    plot_embeddings(embedding_weights, np.arange(A.shape[0]), y)
