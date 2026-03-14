import numpy as np
import tensorflow as tf
import json

from scipy import sparse

from src.app.services.model_type_enum import ModelType

class EllipticSnapshotSingleton:

    _instance = None
    _initialized = False
    _hops = 3

    def __new__(cls, *args):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance
    
    def __init__(self, snapshot_dir):

        if self._initialized:
            return

        # Load Node Features from File.
        self.node_features = np.load(f"{snapshot_dir}/node_features.npy")

        # Load Adjacent Transactions from Files.
        self.adjacent_hops = []

        for hop in range(self._hops):
            self.adjacent_hops.append(
                sparse.load_npz(f"{snapshot_dir}/adjacent_hop_{hop}.npz")
            )

        # Load Transactions by index from File.
        self.transaction_to_index = self._load_data_from_file(f"{snapshot_dir}/transaction_to_index.json")

        self.index_to_transaction = {
            index: transaction for transaction, index in self.transaction_to_index.items()
        }
        
        # Load Confusion Matrix from file.
        self.transductive_confusion_matrix = self._load_data_from_file(f"{snapshot_dir}/transductive_confusion_matrix.json")
        self.inductive_confusion_matrix = self._load_data_from_file(f"{snapshot_dir}/inductive_confusion_matrix.json")

        # Load Evaluation Results from File.
        self.transductive_evaluation_results = self._load_data_from_file(f"{snapshot_dir}/transductive_evaluation_result.json")
        self.inductive_evaluation_results = self._load_data_from_file(f"{snapshot_dir}/inductive_evaluation_result.json")

        # Load Train Results from File.
        self.transductive_train_results = self._load_data_from_file(f"{snapshot_dir}/transductive_train_result.json")
        self.inductive_train_results = self._load_data_from_file(f"{snapshot_dir}/inductive_train_result.json")

        # Load Validation Results from File.
        self.transductive_val_results = self._load_data_from_file(f"{snapshot_dir}/transductive_val_result.json")
        self.inductive_val_results = self._load_data_from_file(f"{snapshot_dir}/inductive_val_result.json")

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
    
    def get_confusion_matrix_by_model_type(self, model_type: ModelType = ModelType.Transductive):
        return self.transductive_confusion_matrix if model_type == ModelType.Transductive else self.inductive_confusion_matrix

    def get_evaluation_by_model_type(self, model_type: ModelType = ModelType.Transductive):
        return self.transductive_evaluation_results if model_type == ModelType.Transductive else self.inductive_evaluation_results

    def get_train_by_model_type(self, model_type: ModelType = ModelType.Transductive):
        return self.transductive_train_results if model_type == ModelType.Transductive else self.inductive_train_results

    def get_validation_by_model_type(self, model_type: ModelType = ModelType.Transductive):
        return self.transductive_val_results if model_type == ModelType.Transductive else self.inductive_val_results

    def _scipy_to_tf_sparse(self, matrix):

        # Convert Sparse Matrix into a Coordinate format.
        coo = matrix.tocoo()

        indices = np.vstack((coo.row, coo.col)).T
        values = coo.data
        shape = coo.shape

        return tf.sparse.SparseTensor(indices, values, shape)

    def _load_data_from_file(self, file_root):

        # f"{snapshot_dir}/transaction_to_index.json"
        # self.transaction_to_index = json.load(f)
        # self.index_to_transaction = {
        #     index: transaction for transaction, index in self.transaction_to_index.items()
        # }

        with open(file_root) as f:
            data_from_file = json.load(f)

        return data_from_file if data_from_file is not None else {}

    