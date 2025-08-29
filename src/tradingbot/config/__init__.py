from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    env: str = Field(default="dev")
    log_level: str = Field(default="INFO")
    log_file: str | None = None
    log_max_bytes: int = Field(default=10 * 1024 * 1024)
    log_backup_count: int = Field(default=5)
    log_json: bool = False
    sentry_dsn: str | None = None

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
    binance_testnet_api_key: str | None = None
    binance_testnet_api_secret: str | None = None

    bybit_api_key: str | None = None
    bybit_api_secret: str | None = None
    bybit_testnet_api_key: str | None = None
    bybit_testnet_api_secret: str | None = None

    okx_api_key: str | None = None
    okx_api_secret: str | None = None
    okx_api_passphrase: str | None = None
    okx_testnet_api_key: str | None = None
    okx_testnet_api_secret: str | None = None
    okx_testnet_api_passphrase: str | None = None

    # Binance Futures (USDT-M) - TESTNET
    binance_futures_api_key: str | None = None
    binance_futures_api_secret: str | None = None
    binance_futures_testnet: bool = False
    binance_futures_leverage: int = 5    # puedes cambiarlo en CLI también
    binance_futures_market: str = "USDT-M"  # informativo

    # Fees (bps) configurable por entorno
    paper_maker_fee_bps: float = 7.5
    paper_taker_fee_bps: float = 7.5

    # Generic broker fees
    maker_fee_bps: float = 7.5
    passive_rebate_bps: float = 0.0

    # Binance Spot
    binance_spot_maker_fee_bps: float = 7.5
    binance_spot_taker_fee_bps: float = 7.5
    binance_spot_testnet_maker_fee_bps: float = 7.5
    binance_spot_testnet_taker_fee_bps: float = 7.5

    # Binance Futuros USD-M
    binance_futures_maker_fee_bps: float = 2.0
    binance_futures_taker_fee_bps: float = 4.0
    binance_futures_testnet_maker_fee_bps: float = 2.0
    binance_futures_testnet_taker_fee_bps: float = 4.0

    # OKX Spot
    okx_spot_maker_fee_bps: float = 8.0
    okx_spot_taker_fee_bps: float = 10.0
    okx_spot_testnet_maker_fee_bps: float = 7.5
    okx_spot_testnet_taker_fee_bps: float = 7.5

    # OKX Futuros USDT-M
    okx_futures_maker_fee_bps: float = 2.0
    okx_futures_taker_fee_bps: float = 5.0
    okx_futures_testnet_maker_fee_bps: float = 2.0
    okx_futures_testnet_taker_fee_bps: float = 5.0

    # Bybit Spot
    bybit_spot_maker_fee_bps: float = 10.0
    bybit_spot_taker_fee_bps: float = 10.0
    bybit_spot_testnet_maker_fee_bps: float = 10.0
    bybit_spot_testnet_taker_fee_bps: float = 10.0

    # Bybit Futuros
    bybit_futures_maker_fee_bps: float = 2.0
    bybit_futures_taker_fee_bps: float = 5.5
    bybit_futures_testnet_maker_fee_bps: float = 2.0
    bybit_futures_testnet_taker_fee_bps: float = 5.5

    # Deribit Spot
    deribit_spot_maker_fee_bps: float = 0.0
    deribit_spot_taker_fee_bps: float = 0.0

    # Deribit Perpetuos BTC/ETH
    deribit_perp_maker_fee_bps: float = 0.0
    deribit_perp_taker_fee_bps: float = 5.0

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
