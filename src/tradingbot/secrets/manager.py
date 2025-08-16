from __future__ import annotations

import os
import logging
from pathlib import Path
from time import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

# Load .env once on import

def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"'))

_load_env()


class _Key:
    """Internal representation of an API key pair."""

    def __init__(self, key: str, secret: str) -> None:
        self.key = key
        self.secret = secret
        self.calls: deque[float] = deque()

    @property
    def ident(self) -> str:
        return self.key[-4:]


class SecretsManager:
    """Loads API keys and rotates them respecting rate limits."""

    def __init__(
        self,
        rate_limit: int = 10,
        per_seconds: int = 60,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.rate_limit = rate_limit
        self.per_seconds = per_seconds
        self.logger = logger or logging.getLogger(__name__)
        self._keys: Dict[str, List[_Key]] = {}
        self._idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Loading
    def _load_keys(self, prefix: str) -> None:
        prefix = prefix.upper()
        keys: List[_Key] = []
        i = 1
        while True:
            k = os.getenv(f"{prefix}_API_KEY_{i}") or os.getenv(f"{prefix}_API_KEY{i}")
            s = os.getenv(f"{prefix}_API_SECRET_{i}") or os.getenv(f"{prefix}_API_SECRET{i}")
            if not k or not s:
                break
            keys.append(_Key(k, s))
            i += 1
        # Fallback to single key
        if not keys:
            k = os.getenv(f"{prefix}_API_KEY")
            s = os.getenv(f"{prefix}_API_SECRET")
            if k and s:
                keys.append(_Key(k, s))
        if not keys:
            raise KeyError(f"no API keys configured for {prefix}")
        self._keys[prefix] = keys
        self._idx[prefix] = 0

    # ------------------------------------------------------------------
    # Public API
    def get_api_credentials(self, prefix: str) -> Tuple[str, str]:
        prefix = prefix.upper()
        if prefix not in self._keys:
            self._load_keys(prefix)
        idx = self._idx[prefix]
        key = self._keys[prefix][idx]
        self._check_rate_limit(prefix, idx, key)
        idx = self._idx[prefix]
        key = self._keys[prefix][idx]
        self._log_usage(prefix, idx, key)
        return key.key, key.secret

    def validate_permissions(
        self,
        exchange,
        required: Iterable[str] = ("read",),
        forbidden: Iterable[str] = ("withdraw",),
    ) -> None:
        from src.tradingbot.utils.secrets import validate_scopes

        validate_scopes(exchange, logger=self.logger, required=required, forbidden=forbidden)

    # ------------------------------------------------------------------
    # Internals
    def _log_usage(self, prefix: str, idx: int, key: _Key) -> None:
        self.logger.info("using %s key #%d id=%s", prefix, idx, key.ident)

    def _check_rate_limit(self, prefix: str, idx: int, key: _Key) -> None:
        now = time()
        q = key.calls
        q.append(now)
        while q and now - q[0] > self.per_seconds:
            q.popleft()
        if len(q) > self.rate_limit and len(self._keys[prefix]) > 1:
            self.logger.info("rate limit hit for %s key #%d", prefix, idx)
            idx = (idx + 1) % len(self._keys[prefix])
            self._idx[prefix] = idx
            key = self._keys[prefix][idx]
            key.calls.append(now)
            self.logger.info("rotated to %s key #%d id=%s", prefix, idx, key.ident)
