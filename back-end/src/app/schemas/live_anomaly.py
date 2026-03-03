from pydantic import BaseModel
from typing import List

class LiveTransactionRequest(BaseModel):
    node_features: List[List[float]]
    adjacent_list: List[List[List[float]]]
    target_node_index: int

class LiveTransactionResponse(BaseModel):
    anomaly_probability: float

