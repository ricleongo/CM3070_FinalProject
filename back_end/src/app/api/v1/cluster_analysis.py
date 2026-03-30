from fastapi import APIRouter
from back_end.src.app.schemas.cluster_analysis import (
    ClusterAnalysisRequest,
    ClusterAnalysisResponse,
)
from back_end.src.app.services.transductive_service import TransductiveScoringService
from back_end.src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
service = TransductiveScoringService(model)

@router.post("/transductive/cluster-analysis", response_model=ClusterAnalysisResponse)
def cluster_analysis(request: ClusterAnalysisRequest):
    """
        Analyze the entire transaction cluster around a transaction.

        Use Case: 
            Is this transaction part of a suspicious network?
    """

    scores = service.get_cluster_analysis(
        request.transaction_id,
        request.hop_depth
    )

    return ClusterAnalysisResponse(scores=scores)


