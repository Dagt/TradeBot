"""Configuration models for TradeBot."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    mode: str = "paper"
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TRADEBOT_")
