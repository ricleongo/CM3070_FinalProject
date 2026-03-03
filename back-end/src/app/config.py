from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "UoL Final Project, Crypto Currency Anomaly Detection Back-End"
    DEBUG: bool = False
    
    # CORS Settings
    # For Angular, you'd typically allow http://localhost:4200
    ALLOWED_ORIGINS: list[str] = ["http://localhost:4200"]
    
    # Environment config
    model_config = SettingsConfigDict(env_file=".env")

@lru_cache
def get_settings():
    return Settings()
