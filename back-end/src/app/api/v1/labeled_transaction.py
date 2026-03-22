from fastapi import APIRouter
from src.app.services.elliptic_service import EllipticService

router = APIRouter()

service = EllipticService()

@router.get("/elliptic/label/{transaction_id}")
def find_fraud_history(transaction_id: int):

    label = service.get_label_by_transaction(
        transaction_id
    )

    return label