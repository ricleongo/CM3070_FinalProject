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
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = K

        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        self.embed_initializer = 'glorot_uniform'

    ### PUBLIC FUNCTIONS ###

    def build(self, input_shape):
        self._set_kernels()

        # Adding dropout layer
        self.dropout = layers.Dropout(self.dropout_rate)


    def call(self, node_features, adjacent_list, training=False):
        """ Keras Layer Forward Propagation Learning Process
        node_features: [features_length, fan_input]
        adjacent_list: list of SparseTensor adj matrices
        """

        output = 0.0

        for hop in range(self.K + 1):

            # 1 Linear transform
            features_by_hop = self.kernels[hop](node_features)
            
            # 2 Apply dropout AFTER linear projection
            features_by_hop = self.dropout(features_by_hop, training=training)

            # 3 Sparse spatial propagation
            spatial_part = self._sparse_propagation(
                adjacent_list[hop],
                features_by_hop
            )

            output += spatial_part

        output = tf.nn.relu(output)

        # 2nd dropout after activation
        output = self.dropout(output, training=training)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "out_dim": self.out_dim,
            "K": self.K,
            "dropout_rate": self.dropout_rate,
            "l2_reg": self.l2_reg
        })

    ### PRIVATE FUNCTIONS ###

    def _set_kernels(self):
        self.kernels = [
            layers.Dense(
                self.out_dim,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(self.l2_reg)                
            )
            for _ in range(self.K + 1)
        ]

    def _sparse_propagation(self, adjacent_sparse, node_features_transformed):
        """
        adjacent_sparse: tf.sparse.SparseTensor [N, N]
        node_features_transformed: [N, out_dim]
        """        
        return tf.sparse.sparse_dense_matmul(adjacent_sparse, node_features_transformed)


