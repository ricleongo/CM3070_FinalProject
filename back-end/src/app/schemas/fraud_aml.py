from pydantic import BaseModel
from typing import List

class TransactionScore(BaseModel):
    transaction_index: int
    fraud_probability: float

class FraudAMLRequest(BaseModel):
    transaction_ids: List[int]

class FraudAMLResponse(BaseModel):
    scores: List[TransactionScore] | None

