from pydantic import BaseModel

class EvaluationMetrics(BaseModel):
    loss: float
    auc: float
    precision: float
    recall: float
    f1: float
    fdr: float
    nrc: float
    far: float

class EvaluationMetricsResponse(BaseModel):
    metrics: EvaluationMetrics | None