from fastapi import APIRouter
from src.app.schemas.network_risk import (
    NetworkRiskRequest,
    NetworkRiskResponse,
)
from src.app.services.transductive_service import TransductiveScoringService
from src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
service = TransductiveScoringService(model)

@router.post("/network/risk", response_model=NetworkRiskResponse)
def detect_network_risk(request: NetworkRiskRequest):
    """
        Evaluate risk propagation across the graph.

        Use Case: A transaction with moderate probability may still be risky if it is connected to multiple illicit nodes

    """
    score = service.score_network_risk(
        transaction_id=request.transaction_id,
        hop_depth=request.hop_depth
    )

    return NetworkRiskResponse(score=score)
