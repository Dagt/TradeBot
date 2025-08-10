from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    env: str = Field(default="dev")
    log_level: str = Field(default="INFO")

    # DB
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_user: str = "postgres"
    pg_password: str = "postgres"
    pg_db: str = "trading"

    # Exchanges
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
