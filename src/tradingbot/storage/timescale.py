import logging
from sqlalchemy import create_engine, text
from ..config import settings

log = logging.getLogger(__name__)

def pg_url() -> str:
    return f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}@{settings.pg_host}:{settings.pg_port}/{settings.pg_db}"

def get_engine():
    return create_engine(pg_url(), pool_pre_ping=True)

def insert_trade(engine, t):
    with engine.begin() as conn:
        conn.execute(text('''
            INSERT INTO market.trades (ts, exchange, symbol, px, qty, side, trade_id)
            VALUES (now(), :exchange, :symbol, :px, :qty, :side, :trade_id)
        '''), dict(exchange=t.exchange, symbol=t.symbol, px=t.price, qty=t.qty, side=t.side, trade_id=None))
