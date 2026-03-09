import numpy as np
import tensorflow as tf

from scipy import sparse

from src.app.schemas.realtime_scoring import RealtimeScoring
from src.app.schemas.simulate_attack import SimulationScore

class InductiveScoringService:

    def __init__(self, model):
        self.model = model

    def score_realtime_transaction(self, transaction_id, hop_depth=2):

        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        transaction_index = elliptic_snapshot.get_index_by_transaction(transaction_id)

        node_features = elliptic_snapshot.get_node_features()
        adjacency_list = elliptic_snapshot.get_adjacent_hops()
        scipy_sparce_adjacent_list = elliptic_snapshot.get_scipy_sparce_adjacent_hops()

        neighborhood = self._extract_local_neighborhood(
            transaction_index,
            scipy_sparce_adjacent_list,
            hop_depth
        )

        neighborhood = [int(neighbor) for neighbor in neighborhood ]

        sub_features = node_features[neighborhood]

        adjacency = scipy_sparce_adjacent_list[1].tocsr()
        
        sub_adjacent = adjacency[neighborhood][:, neighborhood]

        sub_adjacent = elliptic_snapshot.convert_sparse_list_to_tensors(sub_adjacent)

        prediction = self.model(
            sub_features,
            sub_adjacent,
            training=False
        )

        fraud_probability = float(prediction.numpy()[0])

        return RealtimeScoring(
            transaction_id = transaction_id,
            fraud_probability = fraud_probability,
            risk_level = ''
        )
    

    def simulate_attack(self, transaction_features, connected_transactions):

        from src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None

        node_features = elliptic_snapshot.get_node_features()

        # convert transaction ids to indices
        neighbor_indices = [
            elliptic_snapshot.get_index_by_transaction(trx)
            for trx in connected_transactions
        ]

        # build feature matrix
        neighbor_features = node_features[neighbor_indices]

        simulated_features = np.vstack([
            neighbor_features,
            transaction_features
        ])

        num_nodes = simulated_features.shape[0]

        # create adjacency for simulated graph
        sub_adj = sparse.lil_matrix((num_nodes, num_nodes))

        simulated_node_index = num_nodes - 1

        for i in range(len(neighbor_indices)):
            sub_adj[i, simulated_node_index] = 1
            sub_adj[simulated_node_index, i] = 1

        sub_adj = elliptic_snapshot.convert_sparse_list_to_tensors(sub_adj.tocsr())

        prediction = self.model(
            simulated_features,
            sub_adj,
            training=False
        )

        return SimulationScore(
            fraud_probability = float(prediction.numpy()[simulated_node_index])
        )
    

    def _extract_local_neighborhood(self, transaction_index, adjacency_list, hop_depth=2):

        visited = {transaction_index}
        frontier = {transaction_index}

        for hop in range(1, hop_depth + 1):

            adjacency = adjacency_list[hop]

            next_frontier = set()

            for node in frontier:

                neighbors = adjacency.getrow(node).indices
                next_frontier.update(neighbors)

            visited.update(next_frontier)
            frontier = next_frontier

        return list(visited)

    