import numpy as np
import networkx as nx
import pandas as pd

from sklearn.metrics import roc_curve, precision_recall_curve

from src.app.schemas.fraud_history import TransactionScore
from src.app.schemas.network_risk import RiskScore
from src.app.schemas.network_laundering import LaunderingScore
from src.app.schemas.cluster_analysis import ClusterAnalysisScore
from src.app.schemas.network_subgraph import SubGraphNode, SubGraphEdge
from src.app.services.model_type_enum import ModelType
from src.app.schemas.loss_results import LossResults

class TransductiveScoringService:

    def __init__(self, model):
        self.model = model

    def get_top_flagged_transactions(self, top_list):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        transactions_df = pd.read_csv("data/elliptic/elliptic_txs_classes.csv")

        return transactions_df[transactions_df["class"]!="unknown"]["txId"].head(top_list)


    def get_score_history(self, transaction_ids):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        transaction_indexes = [
            elliptic_snapshot.get_index_by_transaction(transaction_id)
            for transaction_id in transaction_ids
        ]
        
        predictions = self._get_flatten_predictions(elliptic_snapshot)

        return [
            TransactionScore(
                transaction_id = elliptic_snapshot.get_transaction_by_index(index), 
                fraud_probability=float(score)
            )
            for index, score in enumerate(predictions)
            if index in transaction_indexes
        ]
    
    def get_score_network_risk(self, transaction_id, hop_depth=1):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None        

        transaction_index = elliptic_snapshot.get_index_by_transaction(transaction_id)

        predictions = self._get_flatten_predictions(elliptic_snapshot)

        own_risk = float(predictions[transaction_index])

        scipy_sparce_adjacent_list = elliptic_snapshot.get_scipy_sparce_adjacent_hops()
        
        adjacency = scipy_sparce_adjacent_list[hop_depth]

        neighbors = adjacency.getrow(transaction_index).indices

        neighbor_scores = predictions[neighbors]

        neighbor_mean = float(neighbor_scores.mean()) if len(neighbor_scores) else 0
        
        neighbor_max = float(neighbor_scores.max()) if len(neighbor_scores) else 0

        suspicious_neighbors = int((neighbor_scores > 0.8).sum())

        return RiskScore(
            transaction_id = transaction_id,
            own_risk = own_risk,
            neighbor_risk_mean = neighbor_mean,
            neighbor_risk_max = neighbor_max,
            suspicious_neighbors = suspicious_neighbors
        )

    def get_cluster_analysis(self, transaction_id, hop_depth=2):
        # Lazy load snapshot that is loaded from application start.
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        transaction_index = elliptic_snapshot.get_index_by_transaction(transaction_id)

        predictions = self._get_flatten_predictions(elliptic_snapshot)

        scipy_sparce_adjacent_list = elliptic_snapshot.get_scipy_sparce_adjacent_hops()

        neighbors = set([transaction_index])

        frontier = {transaction_index}

        for hop in range(1, hop_depth + 1):

            adjacency = scipy_sparce_adjacent_list[hop]

            next_frontier = set()

            for node in frontier:
                new_neighbors = adjacency.getrow(node).indices
                next_frontier.update(new_neighbors)

            neighbors.update(next_frontier)
            frontier = next_frontier

        cluster_transactions = list(neighbors)
    
        cluster_scores = predictions[cluster_transactions]

        return ClusterAnalysisScore(
            transaction_id = transaction_id,
            cluster_size = len(cluster_transactions),
            cluster_risk_mean = float(cluster_scores.mean()),
            cluster_risk_max = float(cluster_scores.max()),
            suspicious_nodes = int((cluster_scores > 0.8).sum())
        )
    
    def find_laundering_networks_by_limit(self, limit=5, risk_threshold=0.8):
        # Lazy load snapshot that is loaded from application start.
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None

        predictions = self._get_flatten_predictions(elliptic_snapshot)

        suspicious_transactions = np.where(predictions > risk_threshold)[0]

        suspicious_graph = nx.Graph()

        suspicious_set = set(suspicious_transactions)

        scipy_sparce_adjacent_list = elliptic_snapshot.get_scipy_sparce_adjacent_hops()

        adjacency = scipy_sparce_adjacent_list[1]

        for suspicious_transaction in suspicious_transactions:

            neighbors = adjacency.getrow(suspicious_transaction).indices

            for n in neighbors:
                if n in suspicious_set:
                    suspicious_graph.add_edge(suspicious_transaction, n)


        # Collecting connected components from suspicious graph.
        clusters = list(nx.connected_components(suspicious_graph))

        results = []

        for cluster_index, cluster in enumerate(clusters):

            cluster_nodes = list(cluster)
            cluster_transaction_ids = [elliptic_snapshot.get_transaction_by_index(node_index) for node_index in cluster_nodes]

            cluster_scores = predictions[cluster_nodes]

            results.append(LaunderingScore(
                cluster_id = elliptic_snapshot.get_transaction_by_index(cluster_index),
                cluster_size = len(cluster_nodes),
                mean_risk = float(cluster_scores.mean()),
                max_risk = float(cluster_scores.max()),
                suspicious_nodes = cluster_transaction_ids #int((cluster_scores > risk_threshold).sum())                
            ))

        results.sort(key=lambda x: x.mean_risk, reverse=True)

        return results[:limit]

    def get_network_subgraph(self, transaction_id, hop_depth=2):

        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None

        transaction_index = elliptic_snapshot.get_index_by_transaction(transaction_id)

        predictions = self._get_flatten_predictions(elliptic_snapshot)


        scipy_sparce_adjacent_list = elliptic_snapshot.get_scipy_sparce_adjacent_hops()

        neighbors = set([transaction_index])
        current_adjacent = { transaction_index }

        for hop in range(1, hop_depth + 1):

            adjacency = scipy_sparce_adjacent_list[hop]

            next_adjacent = set()

            for node in current_adjacent:
                new_neighbors = [int(n) for n in adjacency.getrow(node).indices]

                next_adjacent.update(new_neighbors)

            neighbors.update(next_adjacent) # Adding on top of the stack.

            current_adjacent = next_adjacent # Updating the next adjacent node.

        transaction_indices = list(neighbors) # Converting from dictionary to a list.

        adjacency = scipy_sparce_adjacent_list[1]

        edges = []
        new_neighbors = []

        for node_index in transaction_indices:

            neighbors = [int(n) for n in adjacency.getrow(node_index).indices]

            for neighbor in neighbors:
                # if neighbor in transaction_indices:
                edges.append((node_index, neighbor))
                new_neighbors.append(neighbor)


        [transaction_indices.append(neighbor) for neighbor in new_neighbors]
        transaction_indices = set(transaction_indices)
        transaction_indices = list(transaction_indices)
        
        index_to_transaction_id = elliptic_snapshot.get_transaction_by_index

        node_list = [
            SubGraphNode(
                transaction_id = index_to_transaction_id(node_index),
                risk = float(predictions[node_index])   
            )
            for node_index in transaction_indices
        ]

        edge_list = [
            SubGraphEdge(
                source_transaction_id = index_to_transaction_id(source),
                target_transaction_id = index_to_transaction_id(target)                
            )
            for source, target in edges
        ]

        return {
            "nodes": node_list,
            "edges": edge_list
        }    

    def get_temporal_risk_heatmap(self):
        """
        Returns heatmap matrix: 49 time steps x top N features
        Filtered to illitic-predicted transaction only.
        Cell value = normalized mean feature activation (0-1).
        """
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        predictions = self._get_full_feature_predictions(elliptic_snapshot)

        f1_threshold = self._get_f1_maximizing_threshold(predictions, '1')

        youden_threshold = self._get_youden_statistic_thresshold(predictions, '1')

        print("predictions", predictions.head(2))
        print("f1_threshold, youden_threshold", f1_threshold, youden_threshold)

        illicit_df = predictions[predictions['prediction'] >= f1_threshold]



    def get_model_confusion_matrix(self):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        return elliptic_snapshot.get_confusion_matrix_by_model_type(model_type = ModelType.Transductive)
        
    def get_model_evaluation_results(self):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        return elliptic_snapshot.get_evaluation_by_model_type(model_type = ModelType.Transductive)

    def get_model_train_validation_results(self):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        train_results = elliptic_snapshot.get_train_by_model_type(model_type = ModelType.Transductive)
        val_results = elliptic_snapshot.get_validation_by_model_type(model_type = ModelType.Transductive)
 
        return {
            "train_results": train_results,
            "val_results": val_results
        }

    def _get_flatten_predictions(self, elliptic_snapshot):
        # Collect Elliptic `node_features` Snapshot generated in training step.
        node_features = elliptic_snapshot.get_node_features()
        
        # Collect Elliptic Adjacents by Hop generated in training step.
        adjacent_list = elliptic_snapshot.get_adjacent_hops()

        # Fit with input sample.
        predictions = self.model((node_features, adjacent_list), training=False)

        # Flatten results.
        return predictions.numpy().flatten()
    
    def _get_full_feature_predictions(self, elliptic_snapshot):
        
        features_df = pd.read_csv("data/elliptic/elliptic_txs_features.csv", header=None)
        classes_df = pd.read_csv("data/elliptic/elliptic_txs_classes.csv")

        features_df.columns = (
            ['txId', 'time_step'] +
            [f'local_{i}' for i in range(1, 94)] +
            [f'agg_{j}' for j in range(1, 73)]
        )

        features_df['prediction'] = self._get_flatten_predictions(elliptic_snapshot)
        features_df = features_df[['txId', 'time_step', 'prediction']].copy()

        full_features_df = classes_df.merge(features_df, on='txId')

        # Full Feature results.
        return full_features_df

    def _get_f1_maximizing_threshold(self, dataframe, true_value):

        labelled = dataframe[dataframe['class'] != 'unknown'].copy()

        y_true = np.array(labelled['class'].values)
        y_scores = np.array(labelled['prediction'].values)

        precision, recall, pr_threshold = precision_recall_curve(
            y_true, 
            y_scores,
            pos_label = true_value if true_value is not None else '1'
        )

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

        best_scores = np.argmax(f1_scores)

        best_threshold = pr_threshold[best_scores]

        return best_threshold
    
    
    def _get_youden_statistic_thresshold(self, dataframe, true_value):
        labelled = dataframe[dataframe['class'] != 'unknown'].copy()

        y_true = np.array(labelled['class'].values)
        y_scores = np.array(labelled['prediction'].values)

        fpr, tpr, roc_thresholds = roc_curve(
            y_true,
            y_scores,
            pos_label = true_value if true_value is not None else '1'
        )

        youden_j = tpr - fpr

        best_idx = np.argmax(youden_j)

        best_threshold_roc = roc_thresholds[best_idx]

        return best_threshold_roc

    def weighted_mean(self, group):
        weights = group['illicit_prob']
        return group[top_features].multiply(weights, axis=0).sum() / weights.sum()

