import numpy as np
import tensorflow as tf

class TransductiveScoringService:

    def __init__(self, model):
        self.model = model

    def score_snapshot(self, node_features, adjacent_list):

        X = tf.convert_to_tensor(node_features, dtype=tf.float32)
        A_list = [
            tf.convert_to_tensor(A, dtype=tf.float32)
            for A in adjacent_list
        ]

        predictions = self.model(X, A_list, training=False)
        predictions = predictions.numpy().flatten()

        return [
            {"node_id": idx, "fraud_probability": float(score)}
            for idx, score in enumerate(predictions)
        ]

