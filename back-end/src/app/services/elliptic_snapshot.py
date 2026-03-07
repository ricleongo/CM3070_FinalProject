import numpy as np
import tensorflow as tf
import json

from scipy import sparse


class EllipticSnapshotSingleton:

    _instance = None
    _initialized = False

    def __new__(cls, *args):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance
    
    def __init__(self, snapshot_dir):

        if self._initialized:
            return

        self.node_features = np.load(f"{snapshot_dir}/node_features.npy")

        self.adjacent_hops = []

        self.adjacent_hops.append(
            sparse.load_npz(f"{snapshot_dir}/adjacent_hop_0.npz")
        )

        self.adjacent_hops.append(
            sparse.load_npz(f"{snapshot_dir}/adjacent_hop_1.npz")
        )

        self.adjacent_hops.append(
            sparse.load_npz(f"{snapshot_dir}/adjacent_hop_2.npz") 
        )

        with open(f"{snapshot_dir}/transaction_to_index.json") as f:
            self.transaction_to_index = json.load(f)        

        self._initialized = True

    def get_index_by_transaction(self, transaction_id):
        return self.transaction_to_index.get(str(transaction_id))
    
    def get_node_features(self):
        return self.node_features
    
    def get_adjacent_hops(self):
        return [self._scipy_to_tf_sparse(adjacent) for adjacent in self.adjacent_hops]
    
    def get_scipy_sparce_adjacent_hops(self):
        return self.adjacent_hops
    
    def _scipy_to_tf_sparse(self, matrix):

        # Convert Sparse Matrix into a Coordinate format.
        coo = matrix.tocoo()

        indices = np.vstack((coo.row, coo.col)).T
        values = coo.data
        shape = coo.shape

        return tf.sparse.SparseTensor(indices, values, shape)
    