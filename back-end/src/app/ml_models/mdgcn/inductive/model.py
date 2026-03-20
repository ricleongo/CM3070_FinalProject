import tensorflow as tf

from src.app.ml_models.mdgcn.base_model import BaseSupervisedModel
from src.app.ml_models.mdgcn.inductive.layer import InductiveLayer

@tf.keras.utils.register_keras_serializable()
class SupervisedInductiveModel(BaseSupervisedModel):
    def __init__(self, in_dim, hidden_dim, K, **kwargs):
        super().__init__(**kwargs)

        self.model_name = "Inductive"
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K

        # Add dropout layer.
        self.pre_classifier_dropout = tf.keras.layers.Dropout(0.3)

        # Add Classification Layer.
        self.classifier_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def build(self, input_shape=None):
        super().build(input_shape)

        # Add MD-GCN Layer.
        self.distance_layer = InductiveLayer(
            self.in_dim,
            self.hidden_dim,
            self.K
        )

        # Add Hidden Layer to increase Separation between distance and classifier layers
        self.pre_classifier_dense = tf.keras.layers.Dense(
            32,
            activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4)
        )        

    def call(self, input_values, mask=None, training=False):

        # MD-GCN inductive output
        mdgcn_results = self.distance_layer(input_values)

        # # Appying mask-aware
        # if mask is not None:
        #     mask = tf.cast(mask[:, None], mdgcn_results.dtype)

        #     mdgcn_results = mdgcn_results * mask

        mdgcn_results = self.pre_classifier_dropout(mdgcn_results, training=training)
        mdgcn_results = self.pre_classifier_dense(mdgcn_results, training=False)

        # Classification
        return self.classifier_layer(mdgcn_results)


    def get_config(self):
        config = super().get_config()
        config.update({
            "in_dim": self.in_dim,
            "hidden_dim": self.hidden_dim,
            "K": self.K
        })

        return config
    
    @staticmethod
    def load_model():

        return tf.keras.models.load_model(
            "src/app/ml_models/mdgcn/inductive/model.keras",
            custom_objects={
                "SupervisedInductiveModel": SupervisedInductiveModel
            },
            compile=False
        )

    def save_model(self):
        ROOT_PATH = "../src/app/ml_models/mdgcn"
        
        self.save(ROOT_PATH + "/inductive/model.keras")

    