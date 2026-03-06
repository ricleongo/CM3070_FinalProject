import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class InductiveLayer(layers.Layer):
    
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            K, 
            dropout_rate=0.3,
            l2_reg=1e-4, 
            **kwargs):
        super().__init__(**kwargs)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K

        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        self._set_embeddings()
        self._set_alpha()

    ### PUBLIC FUNCTIONS ###

    def build(self, input_shape):
        self._set_kernels()

        # Adding dropout layer
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, node_features, adjacent_list, training=False):
        return self._get_output(node_features, adjacent_list, training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "K": self.K,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg
        })
        return config

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
        self.feature_kernels = [
            layers.Dense(
                self.out_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
            )
            for _ in range(self.K + 1)
        ]

        self.embedding_kernels = [
            layers.Dense(
                self.out_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
            )
            for _ in range(self.K + 1)
        ]        

    def _get_output(self, node_features, adjacent_list, training=False):
        """
        node_features: node features [N, F]
        adjacent_list: list of adjacency matrices
        """

        # Precompute embeddings
        learned_embeddings = self.embedding_layer(node_features)
        
        output = tf.zeros_like(self.feature_kernels[0](node_features))

        for hop in range(self.K + 1):

            adjacent_sparse = adjacent_list[hop]

            feature_weights = self.feature_kernels[hop](node_features)

            # sparse mathematical multiplication.
            structural = tf.sparse.sparse_dense_matmul(
                adjacent_sparse, 
                feature_weights
            )

            learned = self.embedding_kernels[hop](learned_embeddings)

            output += structural + self.alpha * learned

        # Add dropout before ReLu
        output = self.dropout(output, training=training)

        return tf.nn.relu(output)
