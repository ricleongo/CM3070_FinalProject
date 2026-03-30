from pydantic import BaseModel, computed_field
from typing import List

class SubGraphNode(BaseModel):
    transaction_id: int | None
    risk: float

    @computed_field
    @property
    def risk_level(self) -> str:

        if self.risk < 0.30:
            return "low"

        elif self.risk < 0.60:
            return "medium"

        elif self.risk < 0.85:
            return "high"

        return "critical"    

class SubGraphEdge(BaseModel):
    source_transaction_id: int | None
    target_transaction_id: int | None


class SubGraph(BaseModel):
    nodes: List[SubGraphNode]
    edges: List[SubGraphEdge]


class SubGraphRequest(BaseModel):
    transaction_id: int
    hop_depth: int = 1

class SubGraphResponse(BaseModel):
    subgraph: SubGraph | None
