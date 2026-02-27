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

    def build(self, input_shape):

        # Linear kernels per hop
        self.kernels = [
            layers.Dense(self.out_dim, use_bias=False)
            for _ in range(self.K + 1)
        ]

        # Learnable node embeddings
        self.emb1 = self.add_weight(
            shape=(self.num_nodes, self.embed_dim),
            initializer=self.embed_initializer,
            trainable=True,
            name='embed1'
        )

        self.emb2 = self.add_weight(
            shape=(self.num_nodes, self.embed_dim),
            initializer=self.embed_initializer,
            trainable=True,
            name='embed2'
        )

        # Learnable mixing coefficient
        self.alpha = self.add_weight(
            shape=(),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            name='alpha'
        )

    def call(self, X, adjacent_dist_list):
        """
        X: [N, Fin]
        adjacent_dist_list: list of SparseTensor adj matrices
        """

        out = 0.0

        for k in range(self.K + 1):

            # 1️ Linear transform
            Xk = self.kernels[k](X)

            # 2️ Sparse spatial propagation
            spatial_part = self._sparse_propagation(
                adjacent_dist_list[k],
                Xk
            )

            # 3️ Learned structure propagation
            learned_part = self._learned_propagation(Xk)

            out += spatial_part + self.alpha * learned_part

        return tf.nn.relu(out)

    def _sparse_propagation(self, A_sparse, X_transformed):
        """
        A_sparse: tf.sparse.SparseTensor [N, N]
        X_transformed: [N, out_dim]
        """
        return tf.sparse.sparse_dense_matmul(A_sparse, X_transformed)

    def _learned_propagation(self, X_transformed):
        """
        Efficient learned adjacency propagation:
        Instead of forming A_learn = E1 E2^T,
        compute: E1 (E2^T X)
        """

        temp = tf.matmul(self.emb2, X_transformed, transpose_a=True)

        return tf.matmul(self.emb1, temp)