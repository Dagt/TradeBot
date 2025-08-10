from fastapi import FastAPI
from .. import __init__ as _unused  # ensure package discovery

app = FastAPI(title="TradingBot API")

@app.get("/health")
def health():
    return {"status": "ok"}
