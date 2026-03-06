import numpy as np
import tensorflow as tf
from src.app.schemas.fraud_snapshot import NodeScore

class TransductiveScoringService:

    def __init__(self, model):
        self.model = model

    def score_snapshot(self, node_features, adjacent_list):

        # Convert to Tensor.
        node_features = tf.convert_to_tensor(node_features, dtype=tf.float32)

        # Convert to list of Tensors.
        adjacent_list = [
            tf.convert_to_tensor(adjacent_transaction, dtype=tf.float32)
            for adjacent_transaction in adjacent_list
        ]

        # Fit with input sample.
        predictions = self.model(node_features, adjacent_list, training=False)

        # Flatten results.
        predictions = predictions.numpy().flatten()

        # {"node_id": idx, "fraud_probability": float(score)}

        return [
            NodeScore(node_id= idx, fraud_probability=float(score))
            for idx, score in enumerate(predictions)
        ]

