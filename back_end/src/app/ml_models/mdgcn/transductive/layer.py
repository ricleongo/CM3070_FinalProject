import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
class TransductiveLayer(layers.Layer):
    
    def __init__(
            self, 
            in_dim, 
            out_dim, 
            K,
            dropout_rate=0.5,
            l2_reg=1e-4,
            **kwargs):
        super().__init__(**kwargs)
        
        self.in_dim = in_dim                # Number of input nodes for this Layer.
        self.out_dim = out_dim              # Number of expected nodes to be output for this layer.
        self.max_dist = K                   # Maximum hop distance.
        self.dropout_rate = dropout_rate    # Dropout rate for inter Layer dropout.
        self.l2_reg = l2_reg                # L2 Regularization rate.

        self.embed_initializer = 'glorot_uniform'

    ### PUBLIC FUNCTIONS ###

    def build(self, input_shape):
        # Setting up kernels for node_features.
        self._setup_kernels()

        # Creating dropout layer
        self.dropout = layers.Dropout(self.dropout_rate)


    def call(self, inputs, training=False):
        """ Keras Layer Forward Propagation Learning Process
        node_features: [features_length, fan_input]
        adjacent_list: list of SparseTensor adj matrices
        """

        node_features, adjacent_list = inputs

        output = 0.0

        for hop in range(self.max_dist + 1):

            # this is the `~A` part of the MD-GCN formula "Symmetrically normalized adjacency"
            symetric_A = adjacent_list[hop]

            # Linear transform (this is the `H^l` part of the MD-GCN formula).
            matrix_nodes = self.kernels[hop](node_features)
            
            # Apply dropout to Matrix of Nodes.
            matrix_nodes = self.dropout(matrix_nodes, training=training)

            # Sparse Mathematical multiplication on both previous results.
            matmul_result = tf.sparse.sparse_dense_matmul(
                symetric_A,
                matrix_nodes
            )

            output += matmul_result

        # Applying ReLU Non-Linear Activation Function.
        output = tf.nn.relu(output)

        # 2nd dropout after activation
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

    ### PRIVATE FUNCTIONS ###

    def _setup_kernels(self):
        self.kernels = [
            layers.Dense(
                self.out_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)                
            )
            for _ in range(self.max_dist + 1)
        ]
