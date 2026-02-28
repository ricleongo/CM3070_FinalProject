import tensorflow as tf
from tensorflow.keras import layers

class InductiveMDGCNLayer(layers.Layer):
    
    def __init__(self, in_dim, out_dim, K, **kwargs):
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K

        self._set_embeddings()
        self._set_alpha()

    ### PUBLIC FUNCTIONS ###

    def build(self, input_shape):
        self._set_kernels()

    def call(self, X, adjacent_dist_list):
        return self._get_output(X, adjacent_dist_list)

    ### PRIVATE FUNCTIONS ###

    def _set_embeddings(self):
        self.embedding_layer = layers.Dense(self.out_dim, activation=None, use_bias=False)

    def _set_alpha(self):
        self.alpha = tf.Variable(0.1, trainable=True, dtype=tf.float32)

    def _set_kernels(self):
        self.kernels = [
            layers.Dense(self.out_dim, use_bias=False)
            for _ in range(self.K + 1)
        ]

    def _get_output(self, node_features, adjacent_dist_list):
        """
        node_features: node features [N, F]
        adjacent_dist_list: list of adjacency matrices [N, N] (can be sparse)
        """

        learned_embeddings = self.embedding_layer(node_features)

        output = 0.0
        for hop in range(self.K + 1):
            adjacent_hop = adjacent_dist_list[hop]  # can be sparse tensor
            A_hat = adjacent_hop + self.alpha * tf.matmul(learned_embeddings, learned_embeddings, transpose_b=True)

            # Symmetric normalization
            deg = tf.reduce_sum(A_hat, axis=1)
            deg_inv_sqrt = tf.linalg.diag(tf.pow(deg, -0.5))
            A_norm = tf.matmul(tf.matmul(deg_inv_sqrt, A_hat), deg_inv_sqrt)

            output += tf.matmul(A_norm, self.kernels[hop](node_features))

        return tf.nn.relu(output)
