from fastapi import APIRouter
from .fraud_aml import router as fraud_router
from .live_anomaly import router as live_router
from .network_risk import router as network_risk

api_router = APIRouter()

api_router.include_router(fraud_router, tags=["Fraud Anti Money Laundering"])
api_router.include_router(live_router, tags=["Live Anomaly"])
api_router.include_router(network_risk, tags=["Network Risk"])
