from pydantic import BaseModel, computed_field
from typing import List

class SimulationScore(BaseModel):
    fraud_probability: float

    @computed_field
    @property
    def risk_level(self) -> str:

        if self.fraud_probability < 0.30:
            return "low"

        elif self.fraud_probability < 0.60:
            return "medium"

        elif self.fraud_probability < 0.85:
            return "high"

        return "critical"


class SimulateAttackRequest(BaseModel):
    transaction_features: List[float]
    connected_transactions: List[int]

class SimulateAttackResponse(BaseModel):
    score: SimulationScore | None


