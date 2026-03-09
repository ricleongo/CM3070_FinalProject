from pydantic import BaseModel, computed_field

class RiskScore(BaseModel):
    transaction_id: int
    own_risk: float
    neighbor_risk_mean: float
    neighbor_risk_max: float
    suspicious_neighbors: int

    @computed_field
    @property
    def risk_level(self) -> str:

        if self.own_risk < 0.30:
            return "low"

        elif self.own_risk < 0.60:
            return "medium"

        elif self.own_risk < 0.85:
            return "high"

        return "critical"    


class NetworkRiskRequest(BaseModel):
    transaction_id: int
    hop_depth: int = 1

class NetworkRiskResponse(BaseModel):
    score: RiskScore | None
