import tensorflow as tf
from tensorflow.keras import layers

class InductiveMDGCNLayer(layers.Layer):
    def __init__(self, in_dim, out_dim, K, **kwargs):
        super().__init__(**kwargs)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K

        # Instead of per-node embeddings, use a Dense to compute embeddings from features
        self.embedding_layer = layers.Dense(out_dim, activation=None, use_bias=False)
        self.alpha = tf.Variable(0.1, trainable=True, dtype=tf.float32)

    def build(self, input_shape):
        # K separate kernels for K-hop propagation
        self.kernels = [layers.Dense(self.out_dim, use_bias=False) for _ in range(self.K + 1)]

    def call(self, X, adjacent_dist_list):

        return self._get_output(X, adjacent_dist_list)

    def _get_output(self, X, adjacent_dist_list):
        """
        X: node features [N, F]
        adjacent_dist_list: list of adjacency matrices [N, N] (can be sparse)
        """
        # Compute "inductive embeddings" from features
        learned_embeddings = self.embedding_layer(X)  # shape [N, out_dim]

        out = 0.0
        for hop in range(self.K + 1):
            A_k = adjacent_dist_list[hop]  # can be sparse tensor
            A_hat = A_k + self.alpha * tf.matmul(learned_embeddings, learned_embeddings, transpose_b=True)

            # Symmetric normalization
            deg = tf.reduce_sum(A_hat, axis=1)
            deg_inv_sqrt = tf.linalg.diag(tf.pow(deg, -0.5))
            A_norm = tf.matmul(tf.matmul(deg_inv_sqrt, A_hat), deg_inv_sqrt)

            out += tf.matmul(A_norm, self.kernels[hop](X))

        return tf.nn.relu(out)