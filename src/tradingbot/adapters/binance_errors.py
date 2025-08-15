# src/tradingbot/adapters/binance_errors.py
def parse_binance_error_code(exc) -> int | None:
    """
    Extrae el código de error de una excepción del cliente Binance.
    Devuelve None si no lo detecta (errores de red/timeout).
    """
    try:
        # muchos clientes exponen e.args[0] como dict/json
        msg = getattr(exc, "args", [None])[0]
        if isinstance(msg, dict):
            return int(msg.get("code"))
        if isinstance(msg, str) and '"code":' in msg:
            import json

            # intento rápido
            j = json.loads(msg)
            return int(j.get("code"))
    except Exception:
        pass
    return None
