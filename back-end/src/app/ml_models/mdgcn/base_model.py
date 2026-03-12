from abc import ABC, abstractmethod

import tensorflow as tf

from sklearn.metrics import confusion_matrix


@tf.keras.utils.register_keras_serializable()
class BaseSupervisedModel(tf.keras.Model, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.train_weighted = False
        self.pos_weight = 11.0

    # --------------------------------------------------
    # Abstract forward pass
    # --------------------------------------------------
    @abstractmethod
    def call(self, node_features, adjacent_list, mask=None, training=False):
        pass

    # --------------------------------------------------
    # Custom train step (mask-aware)
    # --------------------------------------------------
    def train_step(self, data):
        (node_features, adjacent_list, mask), labels = data

        with tf.GradientTape() as tape:
            # applying MD-GCN model to node_features and adjacent_list
            predictions = self(node_features, adjacent_list, mask=mask, training=True)
            
            # calculating loss weights based on labels and predictions
            # only masked nodes will be taking care.
            if self.train_weighted:
                loss = self.compute_loss_with_weights(labels, predictions, mask, weights = self.pos_weight)
            else:
                loss = self.compute_loss(labels, predictions, mask)

        # applying (loss / weight)
        gradients = tape.gradient(loss, self.trainable_variables) 

        # applying backpropagation 
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss

    # --------------------------------------------------
    # Custom validation step
    # --------------------------------------------------    
    def val_step(self, data):
        (node_features, adjacent_list, mask), labels = data
        
        # applying MD-GCN model to node_features and adjacent_list
        predictions = self(node_features, adjacent_list, mask=mask, training=False)
        
        # calculating loss weights based on labels and predictions
        # only masked nodes will be taking care.
        if self.train_weighted:
            loss = self.compute_loss_with_weights(labels, predictions, mask, weights = self.pos_weight)
        else:
            loss = self.compute_loss(labels, predictions, mask)

        return loss, predictions


    def fit(self, node_features, adjacent_list, labels, train_mask,
            val_data=None, epochs=100, verbose=1):

        self.train_loss_history = []
        self.validation_loss_history = []

        for epoch in range(epochs + 1):

            # ----- Training step -----
            train_loss = self.train_step(((node_features, adjacent_list, train_mask), labels))

            # ----- Validation step -----
            val_loss = None

            if val_data is not None:
                val_node_features, val_adjacent_list, val_labels, val_mask = val_data

                val_loss, _ = self.val_step(
                    ((val_node_features, val_adjacent_list, val_mask), val_labels)
                )

            if (epoch % 10 == 0 or epochs - epoch == 0):

                self.train_loss_history.append({
                    "name": f"epoch-{epoch}",
                    "value": float(train_loss.numpy())
                })

                self.validation_loss_history.append({
                    "name":  f"epoch-{epoch}",
                    "value": float(val_loss.numpy()) if val_loss is not None else 0.0
                })

                if verbose:
                    print(f"Epoch {epoch}")
                    print("Train Loss:", float(train_loss))

                    if val_loss is not None:
                        print("Val Loss:", float(val_loss))


        self.save_model()


    # --------------------------------------------------
    # Manual graph evaluation
    # --------------------------------------------------
    def evaluate_graph(self, node_features, adjacent_list, labels, mask, threshold=0.5):
        """
        Custom evaluation method for masked nodes, reusing compiled metrics
        """
        results = {}

        # Ensure mask is boolean
        mask = tf.cast(mask, tf.bool)

        # Get predictions
        predictions = self(node_features, adjacent_list, mask=mask, training=False)

        # Apply mask
        y_true = tf.boolean_mask(labels, mask)
        y_pred = tf.boolean_mask(predictions, mask)

        # Apply threshold
        y_pred_binary = tf.cast(y_pred > threshold, y_true.dtype)

        # Setup Confusion Matrix Results
        self.set_confusion_matrix(y_true, y_pred_binary)

        for metric in self.metrics:
            metric.reset_state()

        # Update metrics using binary predictions
        for metric in self.metrics:
            metric.update_state(y_true, y_pred_binary)

        metrics = {metric.name: metric.result() for metric in self.metrics}

        loss = metrics.get("loss", {})
        compile_metrics = metrics.get("compile_metrics", {})
        
        auc = compile_metrics.get("auc", 0.0)
        precision = compile_metrics.get("precision", 0.0)
        recall = compile_metrics.get("recall", 0.0)

        if loss is not None:
            results['loss'] = float(loss)

        if auc is not None:
            results['auc'] = float(auc)
        
        if precision is not None:
            results['precision'] = float(precision)
        
        if recall is not None:
            results['recall'] = float(recall)

        if precision + recall > 0:
            results['f1'] = float(2 * precision * recall / (precision + recall + 1e-8))

        return results
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    def get_train_history(self):
        return self.train_loss_history
    
    def get_validation_history(self):
        return self.validation_loss_history

    def get_confusion_matrix_results(self):
        return self.confusion_matrix_results

    def set_confusion_matrix(self, y_true, y_pred):

        tn, fp, fn, tp = confusion_matrix(
            y_true.numpy(),
            y_pred.numpy()
        ).ravel()

        # Create dictionary
        self.confusion_matrix_results = {
            "TN": int(tn),
            "FP": int(fp), 
            "FN": int(fn), 
            "TP": int(tp)
        }

    def compute_loss(self, labels, predicted, mask):
        """Calculating Loss using normal Binary Cross Entropy function"""

        mask = tf.cast(mask, tf.float32)

        # Binary Cross Entropy
        loss = tf.keras.losses.binary_crossentropy(labels, predicted)

        # Apply mask
        loss = loss * mask
        
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)
    
    def compute_loss_with_weights(self, labels, predicted, mask, weights):
        """Calculating Loss using weighted Binary Cross Entropy manual calculation"""

        mask = tf.cast(mask, tf.float32)

        # Binary Cross Entropy
        loss = tf.keras.losses.binary_crossentropy(labels, predicted)

        # Build class weights
        class_weights = labels * weights + ( 1 - labels )

        # Apply weights
        loss = loss * class_weights
        
        # Apply mask
        loss = loss * mask
        
        return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)