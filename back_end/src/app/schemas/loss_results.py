from pydantic import BaseModel
from typing import List

class LossResults(BaseModel):
    epoch: str
    value: float

class LossResultsResponse(BaseModel):
    train_loss: List[LossResults] | None
    val_loss: List[LossResults] | None
