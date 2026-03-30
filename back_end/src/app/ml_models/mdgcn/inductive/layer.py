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

        self.in_dim = in_dim                # Number of input nodes for this Layer.
        self.out_dim = out_dim              # Number of expected nodes to be output for this layer.
        self.max_dist = K                   # Maximum hop distance.
        self.dropout_rate = dropout_rate    # Dropout rate for inter Layer dropout.
        self.l2_reg = l2_reg                # L2 Regularization rate.

        self._set_embeddings()
        self._set_alpha()

    ### PUBLIC FUNCTIONS ###

    def build(self, input_shape):
        self._set_kernels()

        # Adding dropout layer
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, input_values, training=False):
        """ Keras Layer Forward Propagation Learning Process
        node_features: node features [N, F]
        adjacent_list: list of SparseTensor adjacent matrices
        """

        node_features, adjacent_list = input_values

        # Precompute embeddings
        learned_embeddings = self.embedding_layer(node_features)

        # Compute shared base projection once.
        embedding_base = self.embedding_base_kernel(learned_embeddings)
        
        # Prepopulate output.
        output = tf.zeros_like(self.feature_kernels[0](node_features))

        # Calculate alpha constraint once.
        alpha_constraint = self.alpha_constraint(self.alpha)

        for hop in range(self.max_dist + 1):

            # this is the `~A` part of the MD-GCN formula "Symmetrically normalized adjacency"
            symetric_A = adjacent_list[hop]

            # Linear transform (this is the `H^l` part of the MD-GCN formula).
            matrix_nodes = self.feature_kernels[hop](node_features)

            # Sparse Mathematical multiplication on both previous results.
            matmul_result = tf.sparse.sparse_dense_matmul(
                symetric_A, 
                matrix_nodes
            )

            learned = embedding_base + self.embedding_residual_kernels[hop](learned_embeddings)
            
            output += matmul_result + alpha_constraint[hop] * learned

        # Applying ReLU Non-Linear Activation Function.
        output = tf.nn.relu(output)

        # Adding dropout after activation
        output = self.dropout(output, training=training)

        return output


    def get_config(self):
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "K": self.max_dist,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg
        })
        return config


    ### PRIVATE FUNCTIONS ###

    def _set_embeddings(self):
        self.embedding_layer = tf.keras.Sequential([
            layers.Dense(self.out_dim * 2, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)),
            layers.Dense(self.out_dim, activation=None, kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg))
        ])
        
        
        # layers.Dense(
        #     self.out_dim,
        #     activation=None, 
        #     use_bias=False,
        #     kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        # )

    def _set_alpha(self):
        self.alpha = tf.Variable(
            initial_value=tf.constant([0.1] * (self.max_dist + 1), dtype=tf.float32),
            trainable=True,
        )

        self.alpha_constraint = lambda x: tf.clip_by_value(x, 0.0, 1.0)
        
        # tf.Variable(
        #     initial_value=tf.constant(0.1, dtype=tf.float32),
        #     trainable=True,
        #     constraint=lambda x: tf.clip_by_value(x, 0.0, 1.0)
        # )

    def _set_kernels(self):
        self.feature_kernels = [
            layers.Dense(
                self.out_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
            )
            for _ in range(self.max_dist + 1)
        ]

        self.embedding_base_kernel = layers.Dense(
            self.out_dim,
            use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        )

        self.embedding_residual_kernels = [
            layers.Dense(
                self.out_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
            )
            for _ in range(self.max_dist + 1)
        ]        


        # self.embedding_kernels = [
        #     layers.Dense(
        #         self.out_dim,
        #         use_bias=False,
        #         kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)
        #     )
        #     for _ in range(self.max_dist + 1)
        # ]

