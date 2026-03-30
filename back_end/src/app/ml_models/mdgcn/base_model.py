from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix

@tf.keras.utils.register_keras_serializable()
class BaseSupervisedModel(tf.keras.Model, ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Internal flags to control weighted train.
        self.train_weighted = False     # If it is false use the standard = `Calculating Loss using Unweighted Binary Cross Entropy`
        self.pos_weight = 11.0          # Default weight rate for loss calculation.

        # ── Palette ──────────────────────────────────────────────────────
        self.BG        = '#ffffff'
        self.PANEL     = '#f1f5f9'
        self.BORDER    = '#cbd5e1'
        self.TEXT_PRI  = '#64748B'
        self.TEXT_SEC  = '#8b949e'
        self.TEXT_DESC = '#000000'
        self.ACCENT    = '#58a6ff'    # blue  – correct predictions
        self.WARN      = '#f85149'    # red   – errors
        self.GOOD      = '#3fb950'    # green – header / positive
        self.GOLD      = '#d29922'    # amber – FN (high-cost misses)        

    # --------------------------------------------------
    # Abstract forward pass
    # --------------------------------------------------
    @abstractmethod
    def call(self, inputs, mask, training=False):
        pass

    # --------------------------------------------------
    # Custom train step (mask-aware)
    # --------------------------------------------------
    def train_step(self, data_inputs):
        (node_features, adjacent_list, mask), labels = data_inputs

        with tf.GradientTape() as tape:
            # applying MD-GCN model to node_features and adjacent_list

            predictions = self((node_features, adjacent_list), mask=mask, training=True)
            
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
        predictions = self((node_features, adjacent_list), mask=mask, training=False)
        
        # calculating loss weights based on labels and predictions
        # only masked nodes will be taking care.
        if self.train_weighted:
            loss = self.compute_loss_with_weights(labels, predictions, mask, weights = self.pos_weight)
        else:
            loss = self.compute_loss(labels, predictions, mask)

        return loss, predictions


    def fit(self, node_features, adjacent_list, labels, train_mask,
            val_data=None, epochs=100, verbose=1, early_stopping_patience=15):

        self.train_loss_history = []
        self.validation_loss_history = []

        # Early stopping state
        best_val_loss = float('inf')
        best_weights  = None
        patience_counter = 0

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

                # --- Early stopping logic ---
                if float(val_loss) < best_val_loss:
                    best_val_loss    = float(val_loss)
                    best_weights     = self.get_weights()   # snapshot best weights
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch} — "
                            f"best val loss: {best_val_loss:.4f} "
                            f"(no improvement for {early_stopping_patience} epochs)")

                    if best_weights is not None:
                        # Restore weights from the best checkpoint
                        self.set_weights(best_weights)

                    break

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
        predictions = self((node_features, adjacent_list), mask=mask, training=False)

        # Apply mask
        y_true = tf.boolean_mask(labels, mask)
        y_pred = tf.boolean_mask(predictions, mask)

        # Squeeze to 1D — predictions are [N,1] from sigmoid classifier
        y_true = tf.squeeze(y_true)
        y_pred = tf.squeeze(y_pred)
        
        # Apply threshold
        y_pred_binary = tf.cast(y_pred > threshold, y_true.dtype)

        # Setup Confusion Matrix Results
        self.set_confusion_matrix(y_true, y_pred_binary)

        # --- Standalone metrics — no Keras container involved ---
        auc_metric       = tf.keras.metrics.AUC(name="auc")
        precision_metric = tf.keras.metrics.Precision(name="precision")
        recall_metric    = tf.keras.metrics.Recall(name="recall")

        # AUC uses raw probabilities, not binary predictions
        auc_metric.update_state(y_true, y_pred)

        # Precision and Recall use binary predictions
        precision_metric.update_state(y_true, y_pred_binary)
        recall_metric.update_state(y_true, y_pred_binary)

        precision = float(precision_metric.result())
        recall    = float(recall_metric.result())
        auc       = float(auc_metric.result())


        # Loss — compute directly, not from metric tracker
        if self.train_weighted:
            loss = self.compute_loss_with_weights(labels, predictions, mask, self.pos_weight)
        else:
            loss = self.compute_loss(labels, predictions, mask)

        results['loss']      = float(loss)
        results['auc']       = auc
        results['precision'] = precision
        results['recall']    = recall

        if precision + recall > 0:
            results['f1'] = float(2 * precision * recall / (precision + recall + 1e-8))

        if self.confusion_matrix_results is not None:
            tn = self.confusion_matrix_results["TN"]
            fp = self.confusion_matrix_results["FP"]
            fn = self.confusion_matrix_results["FN"]
            tp = self.confusion_matrix_results["TP"]

            results['fdr'] = tp / (tp + fn + 1e-8)
            results['nrc'] = (tp + fp) / (tp + fp + fn + tn + 1e-8)

        results['far'] = 1.0 - precision

        self.evaluation_metrics = results

        return results
    
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    def get_evaluation_metrics(self):
        return self.evaluation_metrics
    
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

    def graph_safe_smote_loss(self, labels, predictions, mask, alpha=0.2):
        """
        Augments the loss with interpolated illicit node feature predictions.
        Only operates in feature space — no synthetic graph structure needed.
        """
        mask      = tf.cast(mask, tf.float32)
        labels_1d = tf.squeeze(labels)
        preds_1d  = tf.squeeze(predictions)

        # Isolate illicit node predictions within mask
        illicit_mask = tf.cast(labels_1d * mask, tf.bool)
        illicit_preds = tf.boolean_mask(preds_1d, illicit_mask)
        
        n_illicit = tf.shape(illicit_preds)[0]

        if n_illicit < 2:
            return tf.constant(0.0)   # not enough samples to mix

        # Random shuffle of illicit predictions to create pairs
        shuffle_idx     = tf.random.shuffle(tf.range(n_illicit))
        illicit_shuffled = tf.gather(illicit_preds, shuffle_idx)

        # Interpolation coefficient λ ~ Beta(alpha, alpha)
        lam = tf.cast(
            np.random.beta(alpha, alpha),
            dtype=tf.float32
        )

        # Mixed predictions — interpolated between two illicit nodes
        mixed_preds  = lam * illicit_preds + (1 - lam) * illicit_shuffled

        # Target is always 1.0 (illicit) — we're augmenting the positive class
        mixed_labels = tf.ones_like(mixed_preds)

        mixup_loss = tf.keras.losses.binary_crossentropy(
            mixed_labels[:, None], mixed_preds[:, None]
        )

        return tf.reduce_mean(mixup_loss)

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

        # Compute Loss using Binary Cross Entropy.
        loss = tf.keras.losses.binary_crossentropy(labels, predicted)

        # Squeeze and normalize all tensors before applying weighted calculation.
        loss                  = tf.squeeze(loss)
        labels_squeezed       = tf.squeeze(labels)
        mask                  = tf.squeeze(mask)

        # Build label weights
        label_weights = labels_squeezed * weights + ( 1.0 - labels_squeezed )

        # Apply weights
        loss = loss * label_weights
        
        # Apply mask
        loss = loss * mask
        
        base_loss = tf.reduce_sum(loss) / (tf.reduce_sum(mask) + 1e-8)

        # Add mixup augmentation loss with small weight
        mixup = self.graph_safe_smote_loss(labels, predicted, mask, alpha=0.2)

        return base_loss + 0.1 * mixup

    def get_optimal_threshold(self, node_features, adjacent_list, labels, mask):
        """Find threshold that maximizes F1 on validation data."""

        mask = tf.cast(mask, tf.bool)
        predictions = self((node_features, adjacent_list), mask=mask, training=False)
        
        y_true = tf.squeeze(tf.boolean_mask(labels, mask)).numpy()
        y_pred = tf.squeeze(tf.boolean_mask(predictions, mask)).numpy()
        best_f1 = 0.0
        best_threshold = 0.5

        for threshold in np.arange(0.1, 0.95, 0.01):
            y_binary = (y_pred > threshold).astype(float)
            
            tp = ((y_binary == 1) & (y_true == 1)).sum()
            fp = ((y_binary == 1) & (y_true == 0)).sum()
            fn = ((y_binary == 0) & (y_true == 1)).sum()
            
            precision = tp / (tp + fp + 1e-8)
            recall    = tp / (tp + fn + 1e-8)
            f1        = 2 * precision * recall / (precision + recall + 1e-8)
            
            if f1 > best_f1:
                best_f1        = f1
                best_threshold = threshold
        
        print(f"Optimal threshold: {best_threshold:.2f}  →  F1: {best_f1:.4f}")

        return best_threshold        

    def get_benchmark(self, metrics_type, value):
        benchmarks = {
            0: [
                {"max": 0.50, "label": "Poor", "color": "#dc2626", "description": "Below rule-based AML systems"},
                {"max": 0.70, "label": "Acceptable", "color": "#d97706", "description": "Typical legacy bank AML"},
                {"max": 0.85, "label": "Good", "color": "#16a34a", "description": "Modern ML-based AML range"},
                {"max": 1.00, "label": "Excellent", "color": "#0d9488", "description": "Best-in-class crypto AML"},
            ],
            1: [
                {"max": 0.10, "label": "Excellent", "color": "#0d9488", "description": "Best-in-class commercial tools"},
                {"max": 0.30, "label": "Good", "color": "#16a34a", "description": "Acceptable for pre-screening"},
                {"max": 0.60, "label": "Acceptable", "color": "#d97706", "description": "Requires heavy analyst triage"},
                {"max": 1.00, "label": "Poor", "color": "#dc2626", "description": "Unsustainable — alert fatigue"},
            ],
            2: [
                {"max": 0.02, "label": "Low", "color": "#dc2626", "description": "Missing significant activity"},
                {"max": 0.05, "label": "Targeted", "color": "#16a34a", "description": "Ideal for ~2% illicit networks"},
                {"max": 0.15, "label": "Broad Sweep", "color": "#d97706", "description": "High recall, collateral flags"},
                {"max": 1.00, "label": "Excessive", "color": "#dc2626", "description": "Over-reporting SAR risk"},
            ],
        }

        # Replicates .find() logic:
        return next(b for b in benchmarks[metrics_type] if value <= b["max"])

    def draw_heatmap_from_results(self):
        if self.confusion_matrix_results is None:
            return None

        # ── Model results ────────────────────────────────────────────────
        TN = self.confusion_matrix_results['TN']
        FP = self.confusion_matrix_results['FP']
        FN = self.confusion_matrix_results['FN']
        TP = self.confusion_matrix_results['TP']

        total   = TN + FP + FN + TP
        matrix  = np.array([[TN, FP], [FN, TP]])

        # ── Derived metrics ──────────────────────────────────────────────
        accuracy    = (TP + TN) / total
        precision   = TP / (TP + FP)
        recall      = TP / (TP + FN)          # sensitivity
        specificity = TN / (TN + FP)
        f1          = 2 * precision * recall / (precision + recall)
        fdr         = TP / (TP + FN + 1e-8)
        far         = 1.0 - precision
        nrc         = (TP + FP) / (TP + FP + FN + TN + 1e-8)

        fig = plt.figure(figsize=(14, 10), facecolor=self.BG)
        gs  = GridSpec(2, 3, figure=fig, left=0.07, right=0.97, top=0.88, bottom=0.10, wspace=0.38, hspace=0.55)
        ax_mat  = fig.add_subplot(gs[:, 0:2])   # main matrix (left, 2 cols wide)
        ax_met  = fig.add_subplot(gs[0, 2])     # metrics bar (top-right)
        ax_dist = fig.add_subplot(gs[1, 2])     # class distribution (bottom-right)

        for ax in [ax_mat, ax_met, ax_dist]:
            ax.set_facecolor(self.PANEL)
            for spine in ax.spines.values():
                spine.set_edgecolor(self.BORDER)

        # ── Main title ───────────────────────────────────────────────────
        fig.text(0.5, 0.95, f'Confusion Matrix  ·  {self.model_name} MD-GCN AML Model',
                ha='center', va='top', fontsize=15, fontweight='bold',
                color=self.TEXT_PRI, fontfamily='monospace')


        # ─────────────────────────────────────────────────────────────────
        # PANEL 1 — Heatmap matrix
        # ─────────────────────────────────────────────────────────────────
        heatmap_labels = [['TN', 'FP'], ['FN', 'TP']]
        descriptions = [
            ['True Negative\n(Legit → Legit)',  'False Positive\n(Legit → Illicit)'],
            ['False Negative\n(Illicit → Legit)', 'True Positive\n(Illicit → Illicit)']
        ]
        cell_colors = [[self.ACCENT, self.WARN], [self.GOLD, self.GOOD]]
        cell_alpha   = [[0.18, 0.55], [0.55, 0.22]]

        n_rows, n_cols = 2, 2
        xs = [0, 1]
        ys = [1, 0]   # row 0 = top (TN/FP), row 1 = bottom (FN/TP)

        for r in range(n_rows):
            for c in range(n_cols):
                val   = matrix[r, c]
                pct   = val / total * 100
                color = cell_colors[r][c]
                alpha = cell_alpha[r][c]

                rect = mpatches.FancyBboxPatch(
                    (xs[c] + 0.04, ys[r] + 0.04),
                    0.92, 0.92,
                    boxstyle='round,pad=0.02',
                    linewidth=1.6 if (r == c) else 0.8,
                    edgecolor=color,
                    facecolor=color,
                    alpha=alpha,
                    transform=ax_mat.transData,
                    zorder=2
                )
                ax_mat.add_patch(rect)

                cx, cy = xs[c] + 0.5, ys[r] + 0.5

                # Tag (TN / FP etc.)
                ax_mat.text(cx, cy + 0.28, heatmap_labels[r][c],
                            ha='center', va='center',
                            fontsize=13, fontweight='bold',
                            color=color, fontfamily='monospace', zorder=3)

                # Count
                ax_mat.text(cx, cy + 0.06, f'{val:,}',
                            ha='center', va='center',
                            fontsize=26, fontweight='black',
                            color=self.TEXT_PRI, fontfamily='monospace', zorder=3)

                # Percentage
                ax_mat.text(cx, cy - 0.16, f'{pct:.2f}%',
                            ha='center', va='center',
                            fontsize=10, color=self.TEXT_SEC,
                            fontfamily='monospace', zorder=3)

                # Description
                ax_mat.text(cx, cy - 0.35, descriptions[r][c],
                            ha='center', va='center',
                            fontsize=7.5, color=self.TEXT_DESC,
                            fontfamily='monospace', zorder=3, linespacing=1.4)

        # Axis heatmap_labels
        ax_mat.set_xlim(0, 2)
        ax_mat.set_ylim(0, 2)
        ax_mat.set_xticks([0.5, 1.5])
        ax_mat.set_xticklabels(['Predicted\nLegit', 'Predicted\nIllicit'],
                                fontsize=10, color=self.TEXT_PRI, fontfamily='monospace')
        ax_mat.set_yticks([0.5, 1.5])
        ax_mat.set_yticklabels(['Actual\nIllicit', 'Actual\nLegit'],
                                fontsize=10, color=self.TEXT_PRI, fontfamily='monospace', rotation=90, va='center')
        ax_mat.tick_params(length=0)

        # Grid lines
        ax_mat.axhline(1, color=self.BORDER, lw=1.2, zorder=1)
        ax_mat.axvline(1, color=self.BORDER, lw=1.2, zorder=1)

        ax_mat.set_title('Prediction vs. Ground Truth', fontsize=10,
                        color=self.TEXT_SEC, fontfamily='monospace', pad=10)



        # ─────────────────────────────────────────────────────────────────
        # PANEL 2 — Key metrics horizontal bars
        # ─────────────────────────────────────────────────────────────────
        bar_metrics = {
            'Accuracy':                 (accuracy,   self.ACCENT),
            'Precision':                (precision,  self.ACCENT),
            'Recall':                   (recall,     self.GOOD),
            'F1 Score':                 (f1,         self.GOOD),
            'Specificity':              (specificity,self.ACCENT),
            'Fraud Rate':               (fdr,        self.get_benchmark(0, fdr)['color']),
            'Fals Alert Rate':          (far,        self.get_benchmark(1, far)['color']),
            'Network Risk':             (nrc,        self.get_benchmark(2, nrc)['color']),
        }

        names  = list(bar_metrics.keys())
        vals   = [v[0] for v in bar_metrics.values()]
        colors = [v[1] for v in bar_metrics.values()]
        y_pos  = [0, 1, 2, 3, 4, 5, 6.2, 7.4]

        ax_met.barh(y_pos, vals, color=colors, alpha=0.75, height=0.6,
                    edgecolor=[c for c in colors], linewidth=0.8)

        for i, (v, c) in enumerate(zip(vals, colors)):

            if i >= 5:
                benchmark = self.get_benchmark(i-5, v)
                prefix =  f' | {benchmark['label']}'
            else:
                prefix = ''

            ax_met.text(min(v + 0.01, 1.02), y_pos[i], f'{v:.4f}{prefix}',
                        va='center', ha='left', fontsize=7.5,
                        color=c, fontfamily='monospace', fontweight='bold')

            if i >= 5:
                ax_met.text(0.001, y_pos[i] + 0.5, f'{benchmark['description']}',
                            va='center', ha='left', fontsize=6.5,
                            color=self.TEXT_SEC, fontfamily='monospace', fontweight='bold')
                        
        ax_met.set_yticks(y_pos)
        ax_met.set_yticklabels(names, fontsize=8, color=self.TEXT_SEC, fontfamily='monospace')
        ax_met.set_xlim(0, 1.22)
        ax_met.set_xticks([])
        ax_met.set_title('Key Metrics', fontsize=9, color=self.TEXT_SEC, fontfamily='monospace', pad=8)
        ax_met.axvline(1.0, color=self.BORDER, lw=0.8, ls='--')
        ax_met.tick_params(length=0)
        ax_met.invert_yaxis()


        # ─────────────────────────────────────────────────────────────────
        # PANEL 3 — Class distribution donut
        # ─────────────────────────────────────────────────────────────────
        actual_illicit = TP + FN
        actual_legit   = TN + FP

        wedge_colors = [self.GOOD, self.WARN]
        sizes        = [actual_legit, actual_illicit]
        # wedge_labels = [f'Legit\n{actual_legit:,}', f'Illicit\n{actual_illicit:,}']

        # wedges, texts = 
        ax_dist.pie(
            sizes,
            colors=wedge_colors,
            startangle=90,
            wedgeprops=dict(width=0.45, edgecolor=self.PANEL, linewidth=2),
            textprops=dict(color=self.TEXT_PRI, fontsize=7.5, fontfamily='monospace')
        )

        ax_dist.text(0, 0, f'{actual_illicit/total*100:.1f}%\nillicit',
                    ha='center', va='center', fontsize=8.5,
                    color=self.WARN, fontfamily='monospace', fontweight='bold', linespacing=1.4)

        legend_patches = [
            mpatches.Patch(color=self.GOOD, label=f'Legit  {actual_legit:,}  ({actual_legit/total*100:.1f}%)'),
            mpatches.Patch(color=self.WARN, label=f'Illicit {actual_illicit:,}  ({actual_illicit/total*100:.1f}%)')
        ]

        ax_dist.legend(handles=legend_patches, loc='lower center',
                    bbox_to_anchor=(0.5, -0.22), ncol=1,
                    fontsize=7.5, frameon=False,
                    labelcolor=self.TEXT_SEC,
                    prop={'family': 'monospace', 'size': 7.5})

        ax_dist.set_title('Class Distribution', fontsize=9,
                        color=self.TEXT_SEC, fontfamily='monospace', pad=8)


    def draw_barchar_train_val_results(self):

        if self.train_loss_history is None:
            return None

        epochs       = [train['name'] for train in self.train_loss_history]
        train_loss   = [train['value'] for train in self.train_loss_history]
        val_loss     = [train['value'] for train in self.validation_loss_history]
        
        plt.figure(figsize=(14, 5))
        plt.grid(axis='y', linestyle='-', alpha=0.3)

        plt.bar(epochs, val_loss, color='#eef2ff', width=0.4)

        plt.plot(epochs, train_loss, marker='o', color='#2d6a4f', linewidth=2.5, markersize=8)

        for i, val in enumerate(train_loss):
            plt.text(i, val + 0.1, f'{val:.2f}', 
                    ha='center', va='bottom', 
                    color='white', fontweight='bold',
                    bbox=dict(facecolor='#2d6a4f', edgecolor='none', boxstyle='round,pad=0.3'))

        plt.ylim(0, 4)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().tick_params(axis='both', which='both', length=0) # Hide tick marks

        plt.title('Training Loss per Epoch Comparison', 
          fontsize=16, 
          fontweight='bold', 
          loc='left',
          pad=20,
          color= self.TEXT_PRI)
        
        plt.tight_layout()
        plt.show()
