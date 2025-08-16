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

    # Storage backend: "timescale" or "questdb"
    db_backend: str = Field(default="timescale")

    # QuestDB connection parameters (PostgreSQL wire protocol)
    questdb_host: str = "localhost"
    questdb_port: int = 8812
    questdb_user: str = "admin"
    questdb_password: str = "quest"
    questdb_db: str = "qdb"

    # Exchanges Spot (por si los usas luego)
    binance_api_key: str | None = None
    binance_api_secret: str | None = None
    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None

    # Binance Futures (USDT-M) - TESTNET
    binance_futures_api_key: str | None = None
    binance_futures_api_secret: str | None = None
    binance_futures_testnet: bool = True
    binance_futures_leverage: int = 5    # puedes cambiarlo en CLI también
    binance_futures_market: str = "USDT-M"  # informativo

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
