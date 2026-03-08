import numpy as np
import networkx as nx

from src.app.schemas.fraud_history import TransactionScore
from src.app.schemas.network_risk import RiskScore
from src.app.schemas.network_laundering import LaunderingScore
from src.app.schemas.cluster_analysis import ClusterAnalysisScore

class TransductiveScoringService:

    def __init__(self, model):
        self.model = model

    def score_history(self, transaction_ids):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        transaction_indexes = [
            elliptic_snapshot.get_index_by_transaction(transaction_id)
            for transaction_id in transaction_ids
        ]
        
        predictions = self._get_network_predictions(elliptic_snapshot)

        return [
            TransactionScore(
                transaction_index = index, 
                fraud_probability=float(score)
            )
            for index, score in enumerate(predictions)
            if index in transaction_indexes
        ]
    
    def score_network_risk(self, transaction_id, hop_depth=1):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None        

        transaction_index = elliptic_snapshot.get_index_by_transaction(transaction_id)

        predictions = self._get_network_predictions(elliptic_snapshot)

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

    def analyze_cluster(self, transaction_id, hop_depth=2):
        # Lazy load snapshot that is loaded from application start.
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        transaction_index = elliptic_snapshot.get_index_by_transaction(transaction_id)

        predictions = self._get_network_predictions(elliptic_snapshot)

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

        predictions = self._get_network_predictions(elliptic_snapshot)

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

            cluster_scores = predictions[cluster_nodes]

            results.append(LaunderingScore(
                cluster_id = cluster_index,
                cluster_size = len(cluster_nodes),
                mean_risk = float(cluster_scores.mean()),
                max_risk = float(cluster_scores.max()),
                suspicious_nodes = int((cluster_scores > risk_threshold).sum())                
            ))

        results.sort(key=lambda x: x.mean_risk, reverse=True)

        return results[:limit]

    

    def _get_network_predictions(self, elliptic_snapshot):
        # Collect Elliptic Snapshot generated in training step.
        node_features = elliptic_snapshot.get_node_features()
        
        # Collect Elliptic Adjacents by Hop generated in training step.
        adjacent_list = elliptic_snapshot.get_adjacent_hops()

        # Fit with input sample.
        predictions = self.model(node_features, adjacent_list, training=False)

        # Flatten results.
        return predictions.numpy().flatten()