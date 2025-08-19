# Blueprint del TradeBot

Este documento describe, en lenguaje sencillo, todas las piezas que componen
TradeBot.  Sirve como mapa general para entender cómo ingiere datos, genera
señales y ejecuta operaciones en los exchanges soportados.

## 1. Ingesta de datos

- **Tiempo real**: order book L2, trades, best bid/ask, funding y open interest
  vía WebSocket.
- **Histórico**: descargas REST de OHLCV, trades y snapshots de libro.
- **Adaptadores**: módulos por exchange que normalizan símbolos y datos.

_Ejemplo:_
```bash
python -m tradingbot.cli ingest --venue binance_spot --symbol BTC/USDT --depth 20
```

## 2. Generación de features

Cálculo de indicadores técnicos y señales de microestructura:
RSI, retornos, ATR, canales de Keltner, medias móviles, Order Flow Imbalance
(OFI), profundidad del libro y eventos de liquidez.

## 3. Estrategias disponibles

Cada estrategia puede ejecutarse en modo **paper trading** o real.  Se pasan
parámetros mediante un archivo YAML y todas producen señales `buy`, `sell` o
`flat`.

| Estrategia | Descripción breve | Archivo |
|------------|------------------|---------|
| `momentum` | Sigue la tendencia usando RSI y filtro de OFI. | `strategies/momentum.py` |
| `mean_reversion` | Busca regresos a la media con RSI. | `strategies/mean_reversion.py` |
| `breakout_atr` | Rompimiento de canal de Keltner basado en ATR. | `strategies/breakout_atr.py` |
| `breakout_vol` | Rompimiento por desviación estándar. | `strategies/breakout_vol.py` |
| `order_flow` | Promedio del OFI para detectar presión compradora/vendedora. | `strategies/order_flow.py` |
| `mean_rev_ofi` | Reversión a la media usando z‑score de OFI y volatilidad. | `strategies/mean_rev_ofi.py` |
| `depth_imbalance` | Desequilibrio de profundidad del libro. | `strategies/depth_imbalance.py` |
| `liquidity_events` | Vacíos y gaps de liquidez. | `strategies/liquidity_events.py` |
| `cash_and_carry` | Arbitraje spot vs perp según basis/funding. | `strategies/cash_and_carry.py` |
| `cross_exchange_arbitrage` | Arbitraje entre exchanges. | `strategies/cross_exchange_arbitrage.py` |
| `triangular_arb` | Arbitraje triangular dentro del mismo exchange. | `strategies/arbitrage_triangular.py` |
| `arbitrage` | Plantilla simple para spreads entre dos activos. | `strategies/arbitrage.py` |
| `triple_barrier` | Etiquetado de triple barrera con modelo de ML. | `strategies/triple_barrier.py` |

_Ejemplo:_
```bash
python -m tradingbot.cli paper-run --strategy breakout_atr --symbol BTC/USDT
```

## 4. Gestión de riesgo

- **Sizing** por volatilidad objetivo.
- **PortfolioGuard** y **DailyGuard** para límites por símbolo y pérdidas
  diarias.
- **Kill switch** global para cerrar posiciones.

## 5. Ejecución

Router multi‑exchange con algoritmos **TWAP**, **VWAP** y **POV**.  Soporta
órdenes limit, market, post‑only, IOC/FOK y OCO.

_Ejemplo:_
```bash
python -m tradingbot.cli run-bot --exchange binance --symbol ETH/USDT --algo twap
```

## 6. Persistencia

Almacena trades, order books, fills y PnL en **TimescaleDB** o **QuestDB**.
Los escritores son asíncronos y toleran caídas temporales.

## 7. Backtesting y simulación

- **Vectorizado** con `vectorbt` para exploración rápida.
- **Event‑driven** con simulación de colas L2, fees y slippage.

_Ejemplo:_
```bash
python -m tradingbot.cli backtest --strategy mean_reversion --config cfg.yaml
```

## 8. Monitoreo y panel web

Exposición de métricas Prometheus (`ws_uptime`, `ingest_rate`, etc.) y un panel
FastAPI con secciones para credenciales, monitoreo y bots.  La consola web
permite ejecutar cualquier comando de la CLI.

---

Este blueprint ofrece una visión completa del bot.  Para ver cómo cada punto
se conecta con el código fuente consulta [docs/blueprint_map.md](docs/blueprint_map.md).
