from fastapi import APIRouter
from src.app.schemas.network_subgraph import (
    SubGraphRequest,
    SubGraphResponse,
)
from src.app.services.transductive_service import TransductiveScoringService
from src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
service = TransductiveScoringService(model)

@router.post("/transductive/network/subgraph", response_model=SubGraphResponse)
def network_subgraph(request: SubGraphRequest):
    
    subgraph = service.extract_network_subgraph(
        transaction_id= request.transaction_id,
        hop_depth= request.hop_depth
    )

    return SubGraphResponse(subgraph = subgraph)
