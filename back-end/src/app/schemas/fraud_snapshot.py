from pydantic import BaseModel
from typing import List

class NodeScore(BaseModel):
    node_id: int
    fraud_probability: float

class FraudSnapshotRequest(BaseModel):
    node_features: List[List[float]]
    adjacency_matrices: List[List[List[float]]]  # TODO: check if I can build it internally in the service, instead of requesting the entire list.

class FraudSnapshotResponse(BaseModel):
    scores: List[NodeScore]

