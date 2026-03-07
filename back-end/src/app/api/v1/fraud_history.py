from fastapi import APIRouter
from src.app.schemas.fraud_history import (
    FraudHistoryRequest,
    FraudHistoryResponse,
)
from src.app.services.transductive_service import TransductiveScoringService
from src.app.ml_models.mdgcn.transductive.model import SupervisedTransductiveModel

router = APIRouter()

model = SupervisedTransductiveModel.load_model()
service = TransductiveScoringService(model)

# (AML) = Anti Money Laundering, detecting from a list of transactions.
@router.post("/fraud/history", response_model=FraudHistoryResponse)
def detect_fraud_history(request: FraudHistoryRequest):
    """
        Investigation over the historical transactions for posible fraud

        Use Case: 
            A compliance analyst investigates a set of suspicious transactions identified during an audit.

        Possible Actions:
            - Investigate transaction cluster
            - trace fund flows
            - identify laundering networks
    """

    scores = service.score_aml(
        request.transaction_ids
    )

    return FraudHistoryResponse(scores=scores)
