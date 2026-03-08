from fastapi import APIRouter
from src.app.schemas.network_laundering import (
    NetworkLaunderingResponse
)
from src.app.services.transductive_service import TransductiveScoringService
from src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
service = TransductiveScoringService(model)

@router.get("/network/laundering/top", response_model=NetworkLaunderingResponse)
def find_network_laundering(top_limit: int = 5):
    """
        Find top of network money laundering
    """
    
    score = service.find_laundering_networks_by_limit(
        limit= top_limit
    )

    return NetworkLaunderingResponse(score = score)
