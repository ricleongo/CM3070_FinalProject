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

@router.post("/transductive/history", response_model=FraudHistoryResponse)
def find_fraud_history(request: FraudHistoryRequest):
    """
        Investigation over the historical transactions for posible fraud

        Use Case: 
            A compliance analyst investigates a set of suspicious transactions identified during an audit.

        Possible Actions:
            - Investigate transaction cluster
            - trace fund flows
            - identify laundering networks
    """

    scores = service.get_score_history(
        request.transaction_ids
    )

    return FraudHistoryResponse(scores=scores)


@router.get("/transductive/history/{top_list}", response_model=FraudHistoryResponse)
def find_fraud_history(top_list: int = 5):
    """
        Investigation over the historical transactions for posible fraud

        Use Case: 
            A compliance analyst investigates a set of suspicious transactions identified during an audit.

        Possible Actions:
            - Investigate transaction cluster
            - trace fund flows
            - identify laundering networks
    """

    transaction_list = service.get_top_flagged_transactions(top_list)

    scores = service.get_score_history(
        transaction_list
    )

    return FraudHistoryResponse(scores=scores)
