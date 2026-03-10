from pydantic import BaseModel
from typing import List

class LaunderingScore(BaseModel):
    cluster_id: int
    cluster_size: int
    mean_risk: float
    max_risk: float
    suspicious_nodes: int

class NetworkLaunderingRequest(BaseModel):
    top_limit: int = 5

class NetworkLaunderingResponse(BaseModel):
    score: List[LaunderingScore] | None
