from src.app.schemas.fraud_history import TransactionScore
from src.app.schemas.network_risk import RiskScore

class TransductiveScoringService:

    def __init__(self, model):
        self.model = model

    def score_aml(self, transaction_ids):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        transactions_index = [
            elliptic_snapshot.get_index_by_transaction(transaction_id)
            for transaction_id in transaction_ids
        ]
        
        # Collect Elliptic Snapshot generated in training step.
        node_features = elliptic_snapshot.get_node_features()
        
        # Collect Elliptic Adjacents by Hop generated in training step.
        adjacent_list = elliptic_snapshot.get_adjacent_hops()

        # Fit with input sample.
        predictions = self.model(node_features, adjacent_list, training=False)

        # Flatten results.
        predictions = predictions.numpy().flatten()

        return [
            TransactionScore(transaction_index = idx, fraud_probability=float(score))
            for idx, score in enumerate(predictions)
            if idx in transactions_index
        ]
    
    def score_network_risk(self, transaction_id, hop_depth=1):
        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None        

        node_index = elliptic_snapshot.get_index_by_transaction(transaction_id)

        node_features = elliptic_snapshot.get_node_features()
        adjacent_list = elliptic_snapshot.get_adjacent_hops()
        scipy_sparce_adjacent_list = elliptic_snapshot.get_scipy_sparce_adjacent_hops()

        predictions = self.model(node_features, adjacent_list, training=False)
        predictions = predictions.numpy().flatten()

        own_risk = float(predictions[node_index])

        adjacency = scipy_sparce_adjacent_list[hop_depth]

        neighbors = adjacency.getrow(node_index).indices

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


