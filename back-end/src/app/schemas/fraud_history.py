from pydantic import BaseModel, computed_field
from typing import List

class TransactionScore(BaseModel):
    transaction_id: int
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
    

class FraudHistoryRequest(BaseModel):
    transaction_ids: List[int]

class FraudHistoryResponse(BaseModel):
    scores: List[TransactionScore] | None

