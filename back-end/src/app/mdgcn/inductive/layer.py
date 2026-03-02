import tensorflow as tf
from tensorflow.keras import layers

class InductiveLayer(layers.Layer):
    
    def __init__(self, in_dim, out_dim, K, l2_reg=1e-4, **kwargs):
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K

        self.l2_reg = l2_reg

        self._set_embeddings()
        self._set_alpha()

    ### PUBLIC FUNCTIONS ###

    def build(self, input_shape):
        self._set_kernels()

    def call(self, X, adjacent_dist_list):
        return self._get_output(X, adjacent_dist_list)

    ### PRIVATE FUNCTIONS ###

    def _set_embeddings(self):
        self.embedding_layer = layers.Dense(
            self.out_dim,
            activation=None, 
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )

    def _set_alpha(self):
        self.alpha = tf.Variable(
            0.1,
            trainable=True, 
            dtype=tf.float32,
            constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0)
        )

    def _set_kernels(self):
        self.kernels = [
            layers.Dense(
                self.out_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
            )
            for _ in range(self.K + 1)
        ]

    def _get_output(self, node_features, adjacent_list):
        """
        node_features: node features [N, F]
        adjacent_list: list of adjacency matrices [N, N] (can be sparse)
        """

        learned_embeddings = self.embedding_layer(node_features)
        output = 0.0

        for hop in range(self.K + 1):

            adjacent_sparse = adjacent_list[hop]

            feature_weights = self.kernels[hop](node_features)

            # 1 Structural multiplication sparse safe
            structural = tf.sparse.sparse_dense_matmul(adjacent_sparse, feature_weights)

            # 2 Sparse Learning
            embeddings_weights = tf.matmul(learned_embeddings, feature_weights, transpose_a=True)
            
            learned = tf.matmul(learned_embeddings, embeddings_weights)

            output += structural + self.alpha * learned

        return tf.nn.relu(output)
