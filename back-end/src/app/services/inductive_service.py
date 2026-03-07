import tensorflow as tf

class InductiveScoringService:

    def __init__(self, model):
        self.model = model

    def score_live_transaction(self, node_features, adjacency_list, target_idx):

        # Convert to Tensor.
        node_features = tf.convert_to_tensor(node_features, dtype=tf.float32)

        # Convert to list of Tensors.
        adjacent_list = [
            tf.convert_to_tensor(adjacent_transaction, dtype=tf.float32)
            for adjacent_transaction in adjacency_list
        ]

        # Fit with input sample.
        predictions = self.model(node_features, adjacent_list, training=False)

        # Flatten results.
        predictions = predictions.numpy().flatten()

        # Return predicted results.
        return float(predictions[target_idx])
    