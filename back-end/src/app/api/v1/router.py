from fastapi import APIRouter
from .fraud_history import router as history_router
# from .live_anomaly import router as live_router
from .network_risk import router as network_risk_router
from .cluster_analysis import router as cluster_analysis
from .network_laundering import router as network_laundering
from .network_subgraph import router as network_subgraph

api_router = APIRouter()

api_router.include_router(history_router, tags=["Detecting Fraud By History"])
api_router.include_router(network_risk_router, tags=["Network Risk"])
api_router.include_router(cluster_analysis, tags=["Cluster Analysis"])
api_router.include_router(network_laundering, tags=["Network Money Laundering"])
api_router.include_router(network_subgraph, tags=["Pulling Network Subgraph by Transaction Id"])

# api_router.include_router(live_router, tags=["Live Anomaly"])
