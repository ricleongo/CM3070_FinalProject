from fastapi import APIRouter
from .fraud_history import router as history_router
from .live_anomaly import router as live_router
from .network_risk import router as network_risk_router

api_router = APIRouter()

api_router.include_router(history_router, tags=["Detecting Fraud By History"])
api_router.include_router(network_risk_router, tags=["Network Risk"])

# api_router.include_router(live_router, tags=["Live Anomaly"])
