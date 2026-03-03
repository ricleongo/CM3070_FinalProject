from fastapi import APIRouter
from .fraud_snapshot import router as fraud_router
from .live_anomaly import router as live_router

api_router = APIRouter()

api_router.include_router(fraud_router, tags=["Fraud Snapshot"])
api_router.include_router(live_router, tags=["Live Anomaly"])
