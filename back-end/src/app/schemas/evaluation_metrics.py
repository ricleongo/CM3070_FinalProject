# {"loss":0.7979966402,"auc":0.8408412933,"precision":0.7979966402,"recall":0.7008797526,"f1":0.746291995,"fdr":0.7008797654,"nrc":0.0857551897,"far":0.2020033598}
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