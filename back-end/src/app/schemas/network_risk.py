from pydantic import BaseModel

class RiskScore(BaseModel):
    transaction_id: int
    own_risk: float
    neighbor_risk_mean: float
    neighbor_risk_max: float
    suspicious_neighbors: int


class NetworkRiskRequest(BaseModel):
    transaction_id: int
    hop_depth: int = 1

class NetworkRiskResponse(BaseModel):
    score: RiskScore | None
