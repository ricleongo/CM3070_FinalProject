import tensorflow as tf

class InductiveScoringService:

    def __init__(self, model):
        self.model = model

    def score_live_transaction(self, node_features, adjacency_list, target_idx):

        X = tf.convert_to_tensor(node_features, dtype=tf.float32)
        A_list = [
            tf.convert_to_tensor(A, dtype=tf.float32)
            for A in adjacency_list
        ]

        predictions = self.model(X, A_list, training=False)
        predictions = predictions.numpy().flatten()

        return float(predictions[target_idx])
    