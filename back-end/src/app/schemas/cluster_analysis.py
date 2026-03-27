from pydantic import BaseModel
from typing import List

class ClusterAnalysisScore(BaseModel):
    transaction_id: int
    cluster_size: int
    cluster_risk_mean: float
    cluster_risk_max: float
    suspicious_nodes: int

class ClusterAnalysisRequest(BaseModel):
    transaction_id: int
    hop_depth: int

class ClusterAnalysisResponse(BaseModel):
    scores: ClusterAnalysisScore

