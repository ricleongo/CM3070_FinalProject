from fastapi import APIRouter
from back_end.src.app.schemas.heatmap_risk import (
    HeatmapRiskResponse,
)
from back_end.src.app.services.transductive_service import TransductiveScoringService
from back_end.src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
service = TransductiveScoringService(model)

@router.post("/transductive/heatmap/risk", response_model=HeatmapRiskResponse)
def get_heatmap_temporal_risk():
    """
        Evaluate risk propagation across the graph.

        Use Case: A transaction with moderate probability may still be risky if it is connected to multiple illicit nodes

    """
    score = service.get_temporal_risk_heatmap()

    return HeatmapRiskResponse(score=score)
