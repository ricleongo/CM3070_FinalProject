from fastapi import APIRouter
from src.app.schemas.simulate_attack import (
    SimulateAttackRequest,
    SimulateAttackResponse,
)
from src.app.services.inductive_service import InductiveScoringService
from src.app.ml_models.mdgcn.inductive.model import SupervisedInductiveModel

router = APIRouter()

model = SupervisedInductiveModel.load_model()
service = InductiveScoringService(model)

@router.post("/inductive/simulate-attack")
def simulate_attack(request: SimulateAttackRequest):

    score = service.simulate_attack(
        request.transaction_features,
        request.connected_transactions
    )

    return SimulateAttackResponse(
        score = score
    )    