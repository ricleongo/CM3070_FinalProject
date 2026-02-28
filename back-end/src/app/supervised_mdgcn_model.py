import tensorflow as tf

from src.app.transductive_mdgcn_layer import TransductiveMDGCNLayer

class SupervisedMdgcnModel(tf.keras.Model):
    
    def __init__(self, num_nodes, in_dim, hidden_dim, K):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K
        
        self.classifier_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def build(self):
        self._set_distance_layer()

    def call(self, node_features, adjacent_dist_list):
        return self.classifier_layer(
            self.distance_layer(node_features, adjacent_dist_list)
        )        

    def _set_distance_layer(self):
        self.distance_layer = TransductiveMDGCNLayer(
            self.in_dim,
            self.hidden_dim,
            self.num_nodes,
            self.K
        )