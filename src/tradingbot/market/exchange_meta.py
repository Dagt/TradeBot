# src/tradingbot/market/exchange_meta.py
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

try:
    import ccxt
except Exception:
    ccxt = None

from ..execution.normalize import SymbolRules

@dataclass
class ExchangeMeta:
    name: str
    client: any

    @classmethod
    def binanceusdm_testnet(cls, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        if ccxt is None:
            raise RuntimeError("ccxt no disponible")
        client = ccxt.binanceusdm(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
        client.options["defaultType"] = "future"
        client.set_sandbox_mode(True)
        return cls(name="binanceusdm_testnet", client=client)

    @classmethod
    def binance_spot_testnet(cls, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        if ccxt is None:
            raise RuntimeError("ccxt no disponible")
        client = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
        client.options["defaultType"] = "spot"
        client.set_sandbox_mode(True)
        return cls(name="binance_spot_testnet", client=client)

    @classmethod
    def binance_spot(cls, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        if ccxt is None:
            raise RuntimeError("ccxt no disponible")
        client = ccxt.binance(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
        client.options["defaultType"] = "spot"
        return cls(name="binance_spot", client=client)

    def load_markets(self):
        return self.client.load_markets(True)

    @lru_cache(maxsize=2048)
    def rules_for(self, symbol: str) -> SymbolRules:
        m = self.client.market(symbol)
        pstep = None
        qstep = None
        min_qty = None
        min_notional = None

        # CCXT normalized limits (may be partial)
        limits = m.get("limits") or {}
        amount = limits.get("amount") or {}
        price = limits.get("price") or {}
        cost = limits.get("cost") or {}

        if amount.get("min") is not None:
            min_qty = float(amount["min"])
        if price.get("min") is not None:
            pstep = float(price["min"])
        if cost.get("min") is not None:
            min_notional = float(cost["min"])

        # Raw filters provide authoritative steps/mins
        info = m.get("info") or {}
        for f in info.get("filters", []):
            ft = f.get("filterType")
            if ft == "LOT_SIZE":
                if f.get("stepSize"):
                    qstep = float(f["stepSize"])
                if f.get("minQty"):
                    min_qty = float(f["minQty"])
            elif ft in ("PRICE_FILTER", "PERCENT_PRICE_BY_SIDE"):
                if f.get("tickSize"):
                    pstep = float(f["tickSize"])
            elif ft == "MIN_NOTIONAL":
                # Spot usa minNotional, Futures USDM usa notional
                if f.get("minNotional") is not None:
                    min_notional = float(f["minNotional"])
                elif f.get("notional") is not None:
                    min_notional = float(f["notional"])

        return SymbolRules(price_step=pstep, qty_step=qstep, min_qty=min_qty, min_notional=min_notional)
