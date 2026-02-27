import tensorflow as tf
from src.app.transductive_mdgcn_layer import TransductiveMDGCNLayer

class SupervisedMdgcnModel(tf.keras.Model):

    def __init__(self, num_nodes, in_dim, hidden_dim, K):
        super().__init__()
        
        self.distance_layer = TransductiveMDGCNLayer(in_dim, hidden_dim, num_nodes, K)

        self.classifier_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, node_features, adjacent_dist_list):

        return self.classifier_layer(
            self.distance_layer(node_features, adjacent_dist_list)
        )

