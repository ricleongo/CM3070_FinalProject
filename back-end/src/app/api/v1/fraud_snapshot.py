from fastapi import APIRouter, Depends
from src.app.schemas.fraud_snapshot import (
    FraudSnapshotRequest,
    FraudSnapshotResponse,
)
from src.app.services.transductive_service import TransductiveScoringService
from src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
service = TransductiveScoringService(model)

@router.post("/fraud/snapshot", response_model=FraudSnapshotResponse)
def detect_fraud_snapshot(request: FraudSnapshotRequest):

    scores = service.score_snapshot(
        request.node_features,
        request.adjacency_matrices,
    )

    return FraudSnapshotResponse(scores=scores)
