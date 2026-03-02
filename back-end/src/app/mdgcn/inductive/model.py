import tensorflow as tf

from src.app.mdgcn.base_model import BaseSupervisedModel
from src.app.mdgcn.inductive.layer import InductiveLayer

class SupervisedInductiveModel(BaseSupervisedModel):
    def __init__(self, in_dim, hidden_dim, K):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K

        self.classifier_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def build(self, input_shape=None):
        super().build(input_shape)

        self._set_distance_layer()

    def call(self, node_features, adjacent_dist_list, mask=None, training=False):
        # 1. MD-GCN inductive output
        mdgcn_results = self.distance_layer(node_features, adjacent_dist_list, training=training)

        # 2. Appying mask-aware
        if mask is not None:
            mask = tf.cast(mask[:, None], mdgcn_results.dtype)

            mdgcn_results = mdgcn_results * mask

            # mdgcn_results = mdgcn_results * tf.expand_dims(mask, axis=-1)

        # 3. Classification
        return self.classifier_layer(mdgcn_results)


    def _set_distance_layer(self):
        self.distance_layer = InductiveLayer(
            self.in_dim,
            self.hidden_dim,
            self.K
        )
    