from fastapi import APIRouter
from src.app.schemas.live_anomaly import (
    LiveTransactionRequest,
    LiveTransactionResponse,
)
from src.app.services.inductive_service import InductiveScoringService
from src.app.ml_models.mdgcn.inductive.model import SupervisedInductiveModel

router = APIRouter()

model = SupervisedInductiveModel.load_model()
service = InductiveScoringService(model)


@router.post("/fraud/live", response_model=LiveTransactionResponse)
def detect_live_anomaly(request: LiveTransactionRequest):

    score = service.score_live_transaction(
        request.node_features,
        request.adjacent_list,
        request.target_node_index,
    )

    return LiveTransactionResponse(anomaly_probability=score)
