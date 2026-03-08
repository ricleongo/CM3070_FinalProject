from pydantic import BaseModel
from typing import List

class SubGraphNode(BaseModel):
    transaction_id: int | None
    risk: float

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
