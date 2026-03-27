from fastapi import APIRouter
from src.app.schemas.network_laundering import (
    NetworkLaunderingRequest,
    NetworkLaunderingResponse
)
from src.app.services.transductive_service import TransductiveScoringService
from src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
service = TransductiveScoringService(model)

@router.post("/transductive/network/laundering/top", response_model=NetworkLaunderingResponse)
def find_network_laundering(request: NetworkLaunderingRequest):
    """
        Find top of network money laundering
    """
    
    score = service.find_laundering_networks_by_limit(
        limit= request.top_limit
    )

    return NetworkLaunderingResponse(score = score)
