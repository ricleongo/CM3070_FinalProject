import tensorflow as tf

from src.app.ml_models.mdgcn.base_model import BaseSupervisedModel
from src.app.ml_models.mdgcn.transductive.layer import TransductiveLayer

@tf.keras.utils.register_keras_serializable()
class SupervisedTransductiveModel(BaseSupervisedModel):

    def __init__(self, in_dim, hidden_dim, K, **kwargs):
        super().__init__(**kwargs)
        
        self.model_name = "Transductive"
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K

        self.classifier_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def build(self, input_shape=None):
        super().build(input_shape)

        self._set_distance_layer()

    def call(self, input_values, mask=None, training=False):

        node_features, adjacent_list = input_values

        # full MD-GCN layer trained in distance_layer method
        mdgcn_results = self.distance_layer((node_features, adjacent_list), training=training)

        if mask is not None:
            mask = tf.cast(mask[:, None], mdgcn_results.dtype)
            
            mdgcn_results = mdgcn_results * mask

        supervised_results = self.classifier_layer(mdgcn_results)

        return supervised_results

    def get_config(self):
        config = super().get_config()
        
        config.update({
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "K": self.K
        })

        return config
    
    def _set_distance_layer(self):
        self.distance_layer = TransductiveLayer(
            self.in_dim,
            self.hidden_dim,
            self.K
        )

    @staticmethod
    def load_model():

        return tf.keras.models.load_model(
            "src/app/ml_models/mdgcn/transductive/model.keras",
            custom_objects={
                "SupervisedTransductiveModel": SupervisedTransductiveModel
            },
            compile=False            
        )

    def save_model(self):
        ROOT_PATH = "../src/app/ml_models/mdgcn"

        self.save(ROOT_PATH + "/transductive/model.keras")