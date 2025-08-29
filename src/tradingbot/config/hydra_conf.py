from __future__ import annotations

"""Hydra configuration helpers for TradingBot.

This module registers the base configuration and exposes a helper to
load the default YAML file.  The configuration is divided in the main
sections used throughout the project: exchanges, strategies, backtest
parameters and storage options.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class ExchangesConfig:
    """Credentials and options for the supported exchanges."""

    binance_api_key: str = ""
    binance_api_secret: str = ""


@dataclass
class StrategiesConfig:
    """Default strategy selection and its parameters."""

    default: str = "breakout_atr"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestConfig:
    """CSV paths and runtime options for the backtest."""

    data: str = "data/examples/btcusdt_1m.csv"
    symbol: str = "BTC/USDT"
    strategy: str = "breakout_atr"
    latency: int = 1
    window: int = 120
    min_fill_qty: float = 1e-3


@dataclass
class StorageConfig:
    """Persistence backend used during a backtest."""

    backend: str = "sqlite"
    url: str = "sqlite:///:memory:"


@dataclass
class RiskConfig:
    """Risk management parameters used by strategies/daemon."""

    correlation_threshold: float = 0.8
    returns_window: int = 100


@dataclass
class BalanceConfig:
    """Balance rebalancing parameters."""

    enabled: bool = False
    threshold: float = 0.0
    interval: float = 60.0
    assets: List[str] = field(default_factory=list)


@dataclass
class ExchangeVenueConfig:
    """Per-venue configuration such as fees and tick sizes."""

    market_type: str = "spot"
    maker_fee_bps: float = 10.0
    taker_fee_bps: float = 10.0
    tick_size: float = 0.0
    min_fill_qty: float = 1e-3


@dataclass
class FiltersConfig:
    """Thresholds for pre-signal liquidity checks."""

    max_spread: float = float("inf")
    min_volume: float = 0.0
    max_volatility: float = float("inf")


@dataclass
class AppConfig:
    """Top level application configuration."""

    exchanges: ExchangesConfig = field(default_factory=ExchangesConfig)
    strategies: StrategiesConfig = field(default_factory=StrategiesConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    balance: BalanceConfig = field(default_factory=BalanceConfig)
    filters: FiltersConfig = field(default_factory=FiltersConfig)
    exchange_configs: Dict[str, ExchangeVenueConfig] = field(default_factory=dict)


# Register configuration so Hydra validates the structure
cs = ConfigStore.instance()
cs.store(name="base_config", node=AppConfig)


CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"


def load_config(path: str | None = None):
    """Load a configuration from YAML returning an ``OmegaConf`` object."""

    cfg_file = Path(path) if path else CONFIG_PATH
    return OmegaConf.load(cfg_file)
