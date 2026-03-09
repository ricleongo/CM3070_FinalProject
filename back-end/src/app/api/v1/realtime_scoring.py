from fastapi import APIRouter
from src.app.schemas.realtime_scoring import (
    RealtimeScoringRequest,
    RealtimeScoringResponse,
)
from src.app.services.inductive_service import InductiveScoringService
from src.app.ml_models.mdgcn.inductive.model import SupervisedInductiveModel

router = APIRouter()

model = SupervisedInductiveModel.load_model()
service = InductiveScoringService(model)

@router.post("/fraud/realtime-scoring", response_model=RealtimeScoringResponse)
def detect_live_anomaly(request: RealtimeScoringRequest):

    score = service.score_realtime_transaction(
        request.transaction_id
    )

    return RealtimeScoringResponse(score=score)
