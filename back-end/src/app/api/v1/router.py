from fastapi import APIRouter

from .fraud_history import router as history_router
from .network_risk import router as network_risk_router
from .cluster_analysis import router as cluster_analysis
from .network_laundering import router as network_laundering
from .network_subgraph import router as network_subgraph

from .realtime_scoring import router as realtime_scoring
from .simulate_attack import router as simulate_attack

api_router = APIRouter()

# Transductive Use Cases:
api_router.include_router(history_router, tags=["Detecting Fraud By History"])
api_router.include_router(cluster_analysis, tags=["Cluster Analysis"])
api_router.include_router(network_risk_router, tags=["Network Risk"])
api_router.include_router(network_laundering, tags=["Network Money Laundering"])
api_router.include_router(network_subgraph, tags=["Pulling Network Subgraph by Transaction Id"])

# Inductive Use Cases:
api_router.include_router(realtime_scoring, tags=["Realtime Transaction Scoring"])
api_router.include_router(simulate_attack, tags=["Simulating realtime attack"])
