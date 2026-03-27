from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from contextlib import asynccontextmanager

from .config import get_settings, Settings

from src.app.api.v1.router import api_router
from src.app.services.elliptic_snapshot import EllipticSnapshotSingleton

# Getting App Settings
settings = get_settings()

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

elliptic_snapshot: EllipticSnapshotSingleton | None

@asynccontextmanager
async def lifespan(app: FastAPI):

    global elliptic_snapshot

    elliptic_snapshot = EllipticSnapshotSingleton("data/ml_data")

    yield

    elliptic_snapshot = None


app = FastAPI(title = settings.APP_NAME, lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler) # type: ignore

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root Route
@app.get("/api")
@limiter.limit("5/minute")
async def root(
    request: Request,
    settings: Settings = Depends(get_settings)):
    return {"message": f"Welcome to {settings.APP_NAME}"}

# Health check Route
@app.get("/api/health")
async def health_check():
    return {"status": "online"}

# API/V1 Routes
app.include_router(api_router, prefix="/api/v1")
