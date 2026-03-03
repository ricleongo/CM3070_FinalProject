from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel

from .config import get_settings, Settings

from src.app.api.v1.router import api_router

# Getting App Settings
settings = get_settings()

# Initialize Rate Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(title = settings.APP_NAME)
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

# JSON Schema
class Item(BaseModel):
    name: str
    description: str | None = None
    price: float

# Example Routes
@app.get("/")
# @limiter.limit("5/minute")  # Rate limit: 5 requests per minute
async def root(settings: Settings = Depends(get_settings)):
    return {"message": f"Welcome to {settings.APP_NAME}"}

# Health check Route
@app.get("/health")
async def health_check():
    return {"status": "online"}

app.include_router(api_router, prefix="/api/v1")
