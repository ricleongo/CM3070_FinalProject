from pydantic import BaseModel
from typing import List, Dict

class HeatmapRiskPoint(BaseModel):
    x: float
    y: float

class HeatmapRiskSeries(BaseModel):
    name: str
    data: List[HeatmapRiskPoint]


class HeatmapRisk(BaseModel):
    series: HeatmapRiskSeries
    labels: List[str]
    illicit_counts: Dict[str, float]
    time_steps: List[str]


class HeatmapRiskResponse(BaseModel):
    score: HeatmapRisk | None
