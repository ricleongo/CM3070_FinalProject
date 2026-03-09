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

            self.index_to_transaction = {
                index: transaction for transaction, index in self.transaction_to_index.items()
            }

        self._initialized = True

    def get_index_by_transaction(self, transaction_id):
        return self.transaction_to_index.get(str(transaction_id))
    
    def get_transaction_by_index(self, transaction_index: int):
        return self.index_to_transaction.get(transaction_index)
    
    def get_node_features(self):
        return self.node_features
    
    def get_adjacent_hops(self):
        return [self._scipy_to_tf_sparse(adjacent) for adjacent in self.adjacent_hops]

    def convert_sparse_list_to_tensors(self, sparse_list):
        return [self._scipy_to_tf_sparse(adjacent) for adjacent in sparse_list]

    def get_scipy_sparce_adjacent_hops(self):
        return self.adjacent_hops
    
    def _scipy_to_tf_sparse(self, matrix):

        # Convert Sparse Matrix into a Coordinate format.
        coo = matrix.tocoo()

        indices = np.vstack((coo.row, coo.col)).T
        values = coo.data
        shape = coo.shape

        return tf.sparse.SparseTensor(indices, values, shape)
    