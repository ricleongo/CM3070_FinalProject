import tensorflow as tf
from tensorflow.keras import layers

class TransductiveMDGCNLayer(layers.Layer):
    
    def __init__(self, in_dim, out_dim, num_nodes, K, embed_dim=10, **kwargs):
        super().__init__(**kwargs)
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.K = K
        self.embed_dim = embed_dim
        self.embed_initializer = 'glorot_uniform'

    ### PUBLIC FUNCTIONS ###

    def build(self):
        self._set_kernels()
        self._set_embeddings()
        self._set_alpha()


    def call(self, node_features, adjacent_dist_list):
        """ Keras Layer Forward Propagation Learning Process
        node_features: [features_length, fan_input]
        adjacent_dist_list: list of SparseTensor adj matrices
        """

        output = 0.0

        for hop in range(self.K + 1):

            # 1️ Linear transform
            features_by_hop = self.kernels[hop](node_features)

            # 2️ Sparse spatial propagation
            spatial_part = self._sparse_propagation(
                adjacent_dist_list[hop],
                features_by_hop
            )

            # 3️ Learned structure propagation
            learned_part = self._learned_propagation(features_by_hop)

            output += spatial_part + self.alpha * learned_part

        return tf.nn.relu(output)


    ### PRIVATE FUNCTIONS ###

    def _set_kernels(self):
        self.kernels = [
            layers.Dense(self.out_dim, use_bias=False)
            for _ in range(self.K + 1)
        ]

    def _set_embeddings(self):
    
        # Learnable node embeddings
        self.embed1 = self.add_weight(
            shape=(self.num_nodes, self.embed_dim),
            initializer = self.embed_initializer,
            trainable = True,
            name = 'embed1'
        )

        self.embed2 = self.add_weight(
            shape=(self.num_nodes, self.embed_dim),
            initializer=self.embed_initializer,
            trainable=True,
            name='embed2'
        )

    def _set_alpha(self):
        self.alpha = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name='alpha'
        )

    def _sparse_propagation(self, adjacent_sparse, node_features_transformed):
        """
        adjacent_sparse: tf.sparse.SparseTensor [N, N]
        node_features_transformed: [N, out_dim]
        """        
        return tf.sparse.sparse_dense_matmul(adjacent_sparse, node_features_transformed)

    def _learned_propagation(self, node_features_transformed):
        """
        Efficient learned adjacency propagation:
        Instead of forming adjacent_learn = E1 @ E2^T,
        compute: E1 @ (E2^T X)
        """

        return tf.matmul(
            self.embed1, 
            tf.matmul(
                self.embed2,
                node_features_transformed,
                transpose_a=True
            )
        )

