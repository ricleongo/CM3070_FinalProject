from fastapi import APIRouter
from back_end.src.app.schemas.simulate_attack import (
    SimulateAttackRequest,
    SimulateAttackResponse,
)
from back_end.src.app.services.inductive_service import InductiveScoringService
from back_end.src.app.ml_models.mdgcn.inductive.model import SupervisedInductiveModel

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