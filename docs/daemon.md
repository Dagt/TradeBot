# Ejecución como daemon

Ejemplos de configuración para ejecutar los _pollers_ de datos de forma
continua.

## systemd

```
[Unit]
Description=TradingBot perp pollers
After=network.target

[Service]
Type=simple
WorkingDirectory=/ruta/a/TradeBot
ExecStart=/usr/bin/python /ruta/a/poll_perp.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Cron

Archivo `poll_perp.py`:

```
import asyncio
from tradingbot.adapters.binance_futures import BinanceFuturesAdapter
from tradingbot.data.ingestion import poll_perp_metrics

asyncio.run(poll_perp_metrics(BinanceFuturesAdapter(), "BTC/USDT"))
```

Job de cron que ejecuta cada minuto:

```
* * * * * cd /ruta/a/TradeBot && /usr/bin/python /ruta/a/poll_perp.py >> /var/log/tradingbot.log 2>&1
```

Para conocer características avanzadas como estrategias de arbitraje,
panel web y API de control, consulta
[extra_features.md](extra_features.md).
