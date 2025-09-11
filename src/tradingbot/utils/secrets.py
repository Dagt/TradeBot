from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Iterable

# Cargar variables desde un archivo .env local si existe

def _load_env_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip().strip('"')
    for k, v in data.items():
        os.environ.setdefault(k, v)
    return data


def _write_env_file(path: Path, data: Dict[str, str]) -> None:
    """Persist dictionary to ``path`` in ``KEY=VALUE`` format."""
    if not data:
        path.write_text("")
        return
    content = "\n".join(f"{k}={v}" for k, v in data.items()) + "\n"
    path.write_text(content)


def read_secret(key: str, path: Path = Path(".env")) -> Optional[str]:
    """Read a secret value from ``path`` without falling back to ~/.secrets."""
    data = _load_env_file(path)
    return data.get(key)


def set_secret(key: str, value: str, path: Path = Path(".env")) -> None:
    """Store ``key=value`` in ``path`` and update ``os.environ``."""
    data = _load_env_file(path)
    data[key] = value
    _write_env_file(path, data)
    os.environ[key] = value


def delete_secret(key: str, path: Path = Path(".env")) -> None:
    """Remove ``key`` from ``path`` and ``os.environ`` if present."""
    data = _load_env_file(path)
    if key in data:
        del data[key]
        _write_env_file(path, data)
    os.environ.pop(key, None)

# Cargar .env al importar el módulo
_load_env_file(Path(".env"))

_cached_secrets: Dict[str, str] | None = None


def _load_secrets_file() -> Dict[str, str]:
    global _cached_secrets
    if _cached_secrets is not None:
        return _cached_secrets
    path = Path("~/.secrets").expanduser()
    secrets: Dict[str, str] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            secrets[k.strip()] = v.strip().strip('"')
    _cached_secrets = secrets
    return secrets


def get_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    """Obtiene un secreto desde variables de entorno, .env o ~/.secrets."""
    val = os.getenv(key)
    if val is not None:
        return val
    secrets = _load_secrets_file()
    return secrets.get(key, default)


def get_api_credentials(prefix: str) -> tuple[Optional[str], Optional[str]]:
    prefix = prefix.upper()
    return get_secret(f"{prefix}_API_KEY"), get_secret(f"{prefix}_API_SECRET")


def validate_scopes(
    exchange,
    logger: Optional[logging.Logger] = None,
    required: Iterable[str] = ("read", "trade"),
    forbidden: Iterable[str] = ("withdraw",),
) -> None:
    """Valida scopes de la API key y registra *warnings* si faltan."""
    log = logger or logging.getLogger(__name__)
    try:
        has_perm = getattr(exchange, "has", {}).get("fetchPermissions")
        if not has_perm or not hasattr(exchange, "fetch_permissions"):
            log.debug(
                "fetch_permissions no soportado para %s",
                getattr(exchange, "id", ""),
            )
            return
        perms = exchange.fetch_permissions()
    except AttributeError:
        log.debug(
            "fetch_permissions no soportado para %s",
            getattr(exchange, "id", ""),
        )
        return
    except Exception as e:  # pragma: no cover - depende del exchange
        log.warning("no se pudieron obtener permisos: %s", e)
        return

    for scope in required:
        if not perms.get(scope):
            log.warning("API key sin scope requerido '%s'", scope)
    for scope in forbidden:
        if perms.get(scope):
            log.warning("API key con scope no recomendado '%s'", scope)
