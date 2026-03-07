from pydantic import BaseModel
from typing import List

class TransactionScore(BaseModel):
    transaction_index: int
    fraud_probability: float

class FraudHistoryRequest(BaseModel):
    transaction_ids: List[int]

class FraudHistoryResponse(BaseModel):
    scores: List[TransactionScore] | None

