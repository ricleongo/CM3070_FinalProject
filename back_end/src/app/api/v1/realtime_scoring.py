from fastapi import APIRouter
from back_end.src.app.schemas.realtime_scoring import (
    RealtimeScoringRequest,
    RealtimeScoringResponse,
)
from back_end.src.app.services.inductive_service import InductiveScoringService
from back_end.src.app.ml_models.mdgcn.inductive.model import SupervisedInductiveModel

router = APIRouter()

model = SupervisedInductiveModel.load_model()
service = InductiveScoringService(model)

@router.post("/inductive/realtime-scoring", response_model=RealtimeScoringResponse)
def detect_live_anomaly(request: RealtimeScoringRequest):

    score = service.score_realtime_transaction(
        request.transaction_id
    )

    return RealtimeScoringResponse(score=score)
