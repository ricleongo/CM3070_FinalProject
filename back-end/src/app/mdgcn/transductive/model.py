import tensorflow as tf

from src.app.mdgcn.base_model import BaseSupervisedModel
from src.app.mdgcn.transductive.layer import TransductiveLayer

class SupervisedTransductiveModel(BaseSupervisedModel):

    def __init__(self, in_dim, hidden_dim, K):
        super().__init__()
        
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.K = K

        self.classifier_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def build(self, input_shape=None):
        super().build(input_shape)

        self._set_distance_layer()

    def call(self, node_features, adjacent_list, mask=None, training=False):

        # full MD-GCN layer trained in distance_layer method
        mdgcn_results = self.distance_layer(node_features, adjacent_list, training=training)

        if mask is not None:
            mask = tf.cast(mask[:, None], mdgcn_results.dtype)
            
            mdgcn_results = mdgcn_results * mask

        supervised_results = self.classifier_layer(mdgcn_results)

        return supervised_results
    
    def _set_distance_layer(self):
        self.distance_layer = TransductiveLayer(
            self.in_dim,
            self.hidden_dim,
            self.K
        )
    

# import tensorflow as tf

# from src.app.mdgcn.transductive.layer import TransductiveLayer

# class SupervisedTransductiveModel(tf.keras.Model):
    
#     def __init__(self, num_nodes, in_dim, hidden_dim, K):
#         super().__init__()

#         self.num_nodes = num_nodes
#         self.in_dim = in_dim
#         self.hidden_dim = hidden_dim
#         self.K = K
        
#         self.classifier_layer = tf.keras.layers.Dense(1, activation='sigmoid')

#     def build(self, input_shape=None):
#         super().build(input_shape)

#         self._set_distance_layer()

#     def call(self, node_features, adjacent_dist_list, mask=None, training=False):
#         """
#         Forward pass of the model.
        
#         Args:
#             node_features: Tensor of shape [num_nodes, feature_dim]
#             adjacent_dist_list: list of adjacency matrices / distance info
#             mask: Tensor of shape [num_nodes], 1 for active nodes, 0 for masked
#             training: Boolean, True if training

#         Returns:
#             supervised_results: Tensor of shape [num_nodes, 1]
#         """

#         # full MD-GCN layer trained in distance_layer method
#         mdgcn_results = self.distance_layer(node_features, adjacent_dist_list, training=training)

#         if mask is not None:
#             mask = tf.cast(mask[:, None], mdgcn_results.dtype)
            
#             mdgcn_results = mdgcn_results * mask

#         supervised_results = self.classifier_layer(mdgcn_results)

#         return supervised_results
    

#     def train_step(self, data):
#         (node_features, adjacent_list, mask), labels = data

#         with tf.GradientTape() as tape:
#             # applying MD-GCN model to node_features and adjacent_list
#             predictions = self(node_features, adjacent_list, mask=mask, training=True)
            
#             # calculating loss weights based on labels and predictions
#             # only masked nodes will be taking care.
#             loss = self._compute_loss(labels, predictions, mask)

#         # applying (loss / weight)
#         gradients = tape.gradient(loss, self.trainable_variables) 

#         # applying backpropagation 
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

#         return loss
    
#     def val_step(self, data):
#         (node_features, adjacent_list, mask), labels = data
        
#         # applying MD-GCN model to node_features and adjacent_list
#         predictions = self(node_features, adjacent_list, mask=mask, training=False)
        
#         # calculating loss weights based on labels and predictions
#         # only masked nodes will be taking care.
#         loss = self._compute_loss(labels, predictions, mask)

#         return loss, predictions

#     def fit(self, node_features, adjacent_dist_list, labels, train_mask,
#             val_data=None, epochs=100, verbose=1):

#         for epoch in range(epochs):

#             # ----- Training step -----
#             train_loss = self.train_step(((node_features, adjacent_dist_list, train_mask), labels))

#             # ----- Validation step -----
#             val_loss = None

#             if val_data is not None:
#                 val_node_features, val_adjacent_list, val_labels, val_mask = val_data

#                 val_loss, _ = self.val_step(
#                     ((val_node_features, val_adjacent_list, val_mask), val_labels)
#                 )

#             if verbose and epoch % 10 == 0:
#                 print(f"Epoch {epoch}")
#                 print("Train Loss:", float(train_loss))
#                 if val_loss is not None:
#                     print("Val Loss:", float(val_loss))



#     def evaluate_graph(self, node_features, adjacent_list, labels, mask):
#         """
#         Custom evaluation method for masked nodes, reusing compiled metrics
#         """
#         results = {}

#         # Ensure mask is boolean
#         mask = tf.cast(mask, tf.bool)

#         # Get predictions
#         predictions = self(node_features, adjacent_list, mask=mask, training=False)

#         # Apply mask
#         y_true = tf.boolean_mask(labels, mask)
#         y_pred = tf.boolean_mask(predictions, mask)


#         for metric in self.metrics:
#             metric.reset_state()

#         for metric in self.metrics:
#             metric.update_state(y_true, y_pred)

#         metrics = {metric.name: metric.result() for metric in self.metrics}

#         loss = metrics.get("loss", {})
#         compile_metrics = metrics.get("compile_metrics", {})
#         auc = compile_metrics.get("auc", 0.0)
#         precision = compile_metrics.get("precision", 0.0)
#         recall = compile_metrics.get("recall", 0.0)

#         if loss is not None:
#             results['loss'] = float(loss)

#         if auc is not None:
#             results['auc'] = float(auc)
        
#         if precision is not None:
#             results['precision'] = float(precision)
        
#         if recall is not None:
#             results['recall'] = float(recall)

#         if precision + recall > 0:
#             results['f1'] = float(2 * precision * recall / (precision + recall + 1e-8))

#         return results

#     def _set_distance_layer(self):
#         self.distance_layer = TransductiveLayer(
#             self.in_dim,
#             self.hidden_dim,
#             self.K
#         )

#     def _compute_loss(self, labels, predicted, mask):

#         mask = tf.cast(mask, tf.float32)

#         loss = tf.keras.losses.binary_crossentropy(labels, predicted)

#         # Apply mask
#         loss = loss * mask
        
#         return tf.reduce_sum(loss) / tf.reduce_sum(mask)
