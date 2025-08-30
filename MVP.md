# MVP de TradeBot

El objetivo del MVP es ofrecer un recorrido mínimo para ejecutar una
estrategia simple con datos reales en modo de prueba.

## Objetivos

1. Descargar y almacenar datos de mercado.
2. Ejecutar una estrategia en modo "paper" (sin dinero real).
3. Observar métricas básicas en un dashboard.

## Pasos

### 1. Ingestar datos

```bash
python -m tradingbot.cli ingest --symbol BTC/USDT --kind trades --persist
```

Este comando abre un flujo de transacciones del par `BTC/USDT` y lo guarda en
la base de datos.

### 2. Ejecutar una estrategia en modo paper

```bash
python -m tradingbot.cli paper-run --symbol BTC/USDT --strategy breakout_atr
```

El bot aplica la estrategia de rompimiento con ATR sobre el símbolo elegido y
expone métricas en `http://localhost:8000/metrics`. El
`RiskManager` universal (`core/risk_manager.py`) junto con `Account`
(`core/account.py`) dimensiona la posición a partir de un `signal.strength`
continuo y ajusta un trailing stop adaptativo.

Ejemplo de señal:

```python
{"side": "buy", "strength": 0.7, "atr": 3.5}
```

### 3. Revisar métricas

Levante el stack de monitoreo:

```bash
make monitoring-up
```

Abra Grafana (`http://localhost:3000`) y cargue el dashboard "TradeBot
Metrics" para ver PnL, latencia y estado de las estrategias.

Con estos pasos se completa el MVP funcional de TradeBot.
