import numpy as np
import tensorflow as tf

from scipy import sparse

from back_end.src.app.schemas.realtime_scoring import RealtimeScoring
from back_end.src.app.schemas.simulate_attack import SimulationScore
from back_end.src.app.services.model_type_enum import ModelType

class InductiveScoringService:

    def __init__(self, model):
        self.model = model

    def score_realtime_transaction(self, transaction_id, hop_depth=None):
        """
        Score a single transaction in real-time using its local neighborhood.
 
        Extracts the local subgraph around the queried transaction up to
        self.model.K hops, builds a symmetrically normalized hop adjacency
        list matching what InductiveLayer expects, and returns the model's
        fraud probability for that specific transaction node.
 
        Args:
            transaction_id: the transaction identifier to score.
            hop_depth:       number of hops for neighborhood extraction.
                             Defaults to self.model.K to stay consistent
                             with the model's receptive field.
        """
 
        from back_end.src.app.main import elliptic_snapshot
 
        if elliptic_snapshot is None:
            return None
 
        # Tie hop_depth to model.K by default so neighborhood collection
        # and adjacency construction always use the same depth.
        if hop_depth is None:
            hop_depth = self.model.K
 
        transaction_index = elliptic_snapshot.get_index_by_transaction(transaction_id)
 
        node_features              = elliptic_snapshot.get_node_features()
        scipy_sparse_adjacent_list = elliptic_snapshot.get_scipy_sparce_adjacent_hops()
 
        # --- Step 1: collect neighborhood via BFS ---
        neighborhood = self._extract_local_neighborhood(
            transaction_index,
            scipy_sparse_adjacent_list,
            hop_depth
        )
        neighborhood = [int(n) for n in neighborhood]
 
        # Guarantee transaction_index is always at position 0 so that
        # local_index is deterministic regardless of set ordering.
        if transaction_index in neighborhood:
            neighborhood.remove(transaction_index)
        neighborhood = [transaction_index] + neighborhood
 
        local_index = 0   # transaction is always first
 
        # --- Step 2: build sub-graph feature matrix ---
        sub_features = node_features[neighborhood]
 
        # --- Step 3: build K+1 hop adjacency list for the sub-graph ---
        # Use hop-1 (direct connections) as the base adjacency, sliced
        # to the neighborhood, then compute all K hops from it.
        base_adjacency = scipy_sparse_adjacent_list[1].tocsr()
        sub_adj_csr    = base_adjacency[neighborhood][:, neighborhood]
 
        adjacent_list = self._build_local_hop_adjacency(
            sub_adj_csr,
            self.model.K)

        # --- Step 4: forward pass ---
        prediction = self.model(
            (sub_features, adjacent_list),
            training=False
        )
 
        # prediction shape is [N, 1] — extract scalar for the queried node
        fraud_probability = float(prediction.numpy()[local_index][0])
 
        return RealtimeScoring(
            transaction_id    = transaction_id,
            fraud_probability = fraud_probability,
            risk_level        = ''
        )
    
    def simulate_attack(self, transaction_features, connected_transactions):
        """
        Score a hypothetical (unseen) transaction injected into the graph.
 
        Builds a synthetic subgraph where the simulated transaction is
        connected to all provided connected_transactions. The simulated
        node is always placed last in the feature matrix so its index
        is deterministic.
 
        Args:
            transaction_features:    feature vector [1, F] or [F] for the
                                     simulated transaction.
            connected_transactions:  list of transaction IDs that the
                                     simulated node is connected to.
        """
 
        from back_end.src.app.main import elliptic_snapshot
 
        if elliptic_snapshot is None:
            return None
 
        node_features = elliptic_snapshot.get_node_features()
 
        # --- Step 1: resolve neighbor indices and features ---
        neighbor_indices = [
            elliptic_snapshot.get_index_by_transaction(trx)
            for trx in connected_transactions
        ]
 
        neighbor_features = node_features[neighbor_indices]
 
        # Simulated node is appended last — index is always num_nodes - 1
        simulated_features   = np.vstack([neighbor_features, transaction_features])
        num_nodes            = simulated_features.shape[0]
        simulated_node_index = num_nodes - 1
 
        # --- Step 2: build synthetic adjacency ---
        # Simulated node connects bidirectionally to all neighbors.
        # No edges exist between the neighbors themselves in this
        # synthetic graph — only star topology around the new node.
        sub_adj = sparse.lil_matrix((num_nodes, num_nodes))
 
        for i in range(len(neighbor_indices)):
            sub_adj[i, simulated_node_index] = 1
            sub_adj[simulated_node_index, i] = 1
 
        sub_adj_csr = sub_adj.tocsr()
 
        # --- Step 3: build K+1 hop adjacency list ---
        adjacent_list = self._build_local_hop_adjacency(
            sub_adj_csr,
            self.model.K,
            elliptic_snapshot
        )
 
        # --- Step 4: forward pass ---
        prediction = self.model(
            (simulated_features, adjacent_list),
            training=False
        )
 
        # prediction shape is [N, 1] — extract scalar for the simulated node
        fraud_probability = float(prediction.numpy()[simulated_node_index][0])
 
        return SimulationScore(
            fraud_probability=fraud_probability
        )
    

    def get_model_confusion_matrix(self):
        from back_end.src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        return elliptic_snapshot.get_confusion_matrix_by_model_type(model_type = ModelType.Inductive)
        
    def get_model_evaluation_results(self):
        from back_end.src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        return elliptic_snapshot.get_evaluation_by_model_type(model_type = ModelType.Inductive)

    def get_model_train_validation_results(self):
        from back_end.src.app.main import elliptic_snapshot

        if elliptic_snapshot is None:
            return None
        
        train_results = elliptic_snapshot.get_train_by_model_type(model_type = ModelType.Inductive)
        val_results = elliptic_snapshot.get_validation_by_model_type(model_type = ModelType.Inductive)

        return {
            "train_results": train_results,
            "val_results": val_results
        }

    # --------------------------------------------------
    # Private Helpers
    # --------------------------------------------------
 
    def _extract_local_neighborhood(self, transaction_index, adjacency_list, hop_depth=2):
        """
        BFS traversal over pre-computed hop adjacency matrices to collect
        all nodes within hop_depth hops of transaction_index.
 
        Returns an unordered list of node indices (including transaction_index).
        Callers are responsible for enforcing a deterministic ordering.
 
        Args:
            transaction_index: integer index of the root node.
            adjacency_list:    list of scipy sparse matrices, one per hop.
            hop_depth:         number of BFS levels to expand.
        """
 
        visited  = {transaction_index}
        frontier = {transaction_index}
 
        for hop in range(1, hop_depth + 1):
 
            adjacency      = adjacency_list[hop]
            next_frontier  = set()
 
            for node in frontier:
                neighbors = adjacency.getrow(node).indices
                next_frontier.update(neighbors)
 
            visited.update(next_frontier)
            frontier = next_frontier
 
        return list(visited)
 
 
    def _build_local_hop_adjacency(self, sub_adj_csr, K):
        """
        Build a list of K+1 symmetrically normalized hop adjacency matrices
        from a local subgraph CSR matrix.
 
        This replicates the same normalization applied during training on the
        full graph, scoped to the extracted local subgraph.
 
        List structure:
          index 0 → identity (self-connections, hop 0)
          index 1 → normalized A^1 (direct neighbors)
          index k → normalized A^k (k-hop neighbors)
 
        Args:
            sub_adj_csr:      scipy CSR matrix of the local subgraph.
            K:                maximum hop distance (matches model.K).
 
        Returns:
            List of K+1 SparseTensors, one per hop.
        """
 
        n           = sub_adj_csr.shape[0]
        hop_tensors = []
 
        # hop 0: identity — each node aggregates its own features
        identity = sparse.eye(n, format='csr')
        hop_tensors.append(self._csr_to_sparse_tensor(identity))
 
        # hop 1..K: successive powers of the adjacency matrix, each normalized
        A_power = sub_adj_csr.copy()
 
        for k in range(1, K + 1):
 
            if k > 1:
                # Advance one more hop by multiplying with the base adjacency
                A_power = A_power.dot(sub_adj_csr)
 
            # Symmetrically normalize: D^{-0.5} · A^k · D^{-0.5}
            # This matches the Ã_k construction in the MD-GCN paper formula.
            degrees   = np.array(A_power.sum(axis=1)).flatten()
            degrees   = np.where(degrees == 0, 1.0, degrees)   # guard div-by-zero
            d_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
            A_norm     = d_inv_sqrt.dot(A_power).dot(d_inv_sqrt).tocsr()
 
            hop_tensors.append(self._csr_to_sparse_tensor(A_norm))
 
        return hop_tensors    
    

    def _csr_to_sparse_tensor(self, csr_matrix):        
        """
        Convert a scipy CSR matrix directly to a tf.SparseTensor.

        This bypasses elliptic_snapshot.convert_sparse_list_to_tensors()
        which returns raw component tensors rather than an assembled SparseTensor,
        causing 'Input must be a SparseTensor' errors in tf.sparse.sparse_dense_matmul.
        """
        coo     = csr_matrix.tocoo()
        indices = np.column_stack([coo.row, coo.col]).astype(np.int64)
        values  = coo.data.astype(np.float32)
        shape   = list(csr_matrix.shape)

        return tf.SparseTensor(
            indices     = indices,
            values      = values,
            dense_shape = shape
        )    