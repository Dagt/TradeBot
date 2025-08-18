# TradeBot — Diseño Técnico Definitivo (MVP Intradía/Scalping/Arbitraje)

> Documento central que consolida **Blueprint**, **Blueprint Map** y **Definición del TradeBot**. Especifica el diseño técnico completo para un bot local de **trading intradía/scalping** y **arbitraje** sobre **spot** y **perp‑futuros** (Binance/Bybit/OKX), con backtesting serio, ejecución, riesgo, persistencia y monitoreo. Sin planes de tiempo; solo técnica, tecnologías y funcionalidades.

---

## 0) Objetivo y alcance

**Alcance MVP**: operar **spot** y **perp‑futuros** en 2–3 exchanges (Binance, Bybit, OKX) con:

- **Datos en tiempo real** (L1/L2: trades, book, BBA; + funding y OI en perps)
- **Backtesting** (vectorizado y event‑driven con latencia/fees/slippage)
- **Ejecución** (router multi‑exchange, maker/taker, TWAP/VWAP/POV, OCO)
- **Gestión de riesgo** (sizing por volatilidad, límites/correlación, kill‑switch, daily guard)
- **Persistencia** (TimescaleDB/QuestDB)
- **Monitoreo** (métricas Prometheus y panel FastAPI)
- **Paper‑trading** y **testnet**

**No objetivos**: co‑location, HFT <1 ms, opciones complejas.

---

## 1) Arquitectura general

**Capas y flujo**

```
[Exchanges WS/REST] → [Adapters] → [Bus interno asyncio]
                                   ↘ [Persistencia (Timescale/Quest)]
[Feature Eng] → [Estrategias] → [Gestor de Riesgo] → [Router de Ejecución] → [Exchanges]
                                 [Backtester]   [Paper Trading]
                             [Monitoreo + Panel]  [Logs + Métricas]
```

**Tecnologías base**: Python 3.11+, asyncio/uvloop, pydantic, numpy/pandas/numba, polars (opcional), Typer/Click (CLI), ccxt (REST), WS nativos por exchange, vectorbt, FastAPI.

---

## 2) Módulos y responsabilidades

### 2.1 Ingesta de datos

- **WS L1/L2**: trades, book L2 (n≥10), BBA, funding, OI (perps)
- **REST histórico**: OHLCV, trades, snapshots de libro
- **Normalización**: adapters por venue; símbolos unificados
- **Reconexión**: watchdog (ping/pong, backoff exponencial), re‑suscripción

### 2.2 Feature Engineering

- Técnicos: retornos, RSI, **ATR‑EWMA**, **Keltner**, **Donchian**
- Microestructura: **OFI** (Order Flow Imbalance), **Depth/Queue Imbalance**, eventos de liquidez (gaps, vaciado de niveles)
- Eventos perps: **funding rate** y **basis**

### 2.3 Señal / Estrategia

- **Breakout ATR** (canal Keltner)
- **Momentum** intradía (retornos acumulados 1–5–15 m)
- **Mean Reversion** (extremos OFI/volatilidad)
- **Arbitraje**: triangular intra‑exchange, **cross‑exchange**, **cash‑and‑carry/funding‑basis**
- (Fase posterior) ML: triple barrier/meta‑labeling, DSR

### 2.4 Gestión de cartera y riesgo

- **Sizing** por volatilidad objetivo y caps por símbolo/total
- **PortfolioGuard**: hard/soft caps, auto‑close
- **DailyGuard**: pérdida diaria, DD máximo, rachas
- **Kill‑switch** global
- **Correlación intradía**: clusters y cap agregado

### 2.5 Ejecución

- **Router** multi‑exchange (spread–fees–profundidad–latencia)
- **Algoritmos**: TWAP, VWAP, POV
- **Órdenes**: limit, market, **post‑only**, **IOC/FOK**, reduce‑only, iceberg (si venue lo soporta)
- **OCO**: SL/TP, trailing y tiers de TP
- **Slippage**: por participación de volumen y posición en cola
- Validación **minNotional/stepSize**

### 2.6 Persistencia

- **TimescaleDB** (Postgres) o **QuestDB**
- Tablas: `trades`, `orderbook_l2`, `bars_1m`, `funding`, `oi`, `orders`, `fills`, `positions`, `pnl_snapshots`, `risk_events`
- Writer asíncrono con colas y backpressure

### 2.7 Backtesting

- **Vectorizado** (vectorbt) para exploración rápida
- **Event‑driven** propio: simula colas L2, latencia, fees maker/taker, funding, fills parciales
- **Walk‑forward** y reporting (SR/Sortino/Calmar/DSR, hit rate, slippage, maker/taker split)

### 2.8 Monitoreo & Ops

- **Prometheus** exporters: `ws_uptime`, `ingest_rate`, `route_latency`, `order_reject_rate`, `pnl_realized/unrealized`
- **Panel FastAPI**: salud WS, posiciones, órdenes activas, PnL intradía, alertas
- **Alertas**: falta de ticks, DD intradía, tasa de rechazos

---

## 3) Exchanges y conectores

- **Binance**: Spot, Futures USDⓈ‑M/COIN‑M; testnet; funding/basis públicos
- **Bybit**: API v5 unificada; testnet
- **OKX**: REST/WS v5; testnet
- (Opcional) **Deribit**: perps/opciones (funding/basis)

**Conectores**

- REST unificado con **ccxt** (rate limits, idempotencia y reintentos)
- WS **nativo por exchange** para L2, funding y OI (latencia y fiabilidad)

---

## 4) Datos requeridos

**Tiempo real**: `trades`, `orderbook L2`, `BBA`, `funding`, `OI`

**Histórico**: `OHLCV (1s–1h)`, `trades`, `snapshots L2`

**Fuentes**: exchanges (REST/WS), **Kaiko/CoinAPI** (opcional para normalizados)

---

## 5) Señales y estrategias (detallado)

### 5.1 Microestructura

- **OFI** (ej. versión simple en mejor bid/ask):
  - Sea `Δq_bid = max(0, bid_t − bid_{t−1})`, `Δq_ask = max(0, ask_t − ask_{t−1})`
  - **OFI\_t = Δq\_bid − Δq\_ask**
  - Señal: z‑score(OFI) > umbral → prob. de uptick; < −umbral → downtick
- **Depth/Queue Imbalance**: `I = (Σ_qty_bid − Σ_qty_ask)/(Σ_qty_bid + Σ_qty_ask)` en n niveles
- **Eventos de liquidez**: detectar barridos (drops súbitos de profundidad) y gaps

### 5.2 Momentum/Breakouts

- **Keltner** (EMA n=20, ATR n=14): canal superior/inferior = `EMA ± k·ATR`
- **Señal Breakout**: cierre cruza canal superior → long; cruza inferior → short/flat

### 5.3 Mean Reversion rápida

- Tras extremos de **OFI** o spikes de volatilidad; filtros por spread efectivo y latencia

### 5.4 Arbitrajes

- **Triangular** (intra‑exchange): rutas A/B/C con edge neto > 0 tras fees/slippage
- **Cross‑exchange**: misma paridad en 2+ venues; requiere balance y latencia controlada
- **Cash‑and‑carry / Funding‑basis** (perps):
  - Si funding esperado `f` y carry spot‑perp `c`, operar long/short para capturar `c + f − fees − borrow` (ajustar por rollover/latencias)

*(Todos los módulos deben inyectar fees maker/taker por tier y venue)*

---

## 6) Riesgo, PnL y OCO (implementación de referencia)

### 6.1 RiskManager

- Mantiene **posición interna** (`qty`) por símbolo
- Interfaces: `set_position(qty)`, `add_fill(side, qty)`, `size(side, strength)`

### 6.2 PortfolioGuard

- **Hard caps**: `per_symbol_cap_usdt`, `total_cap_usdt` → bloquea
- **Soft caps**: exceso controlado `soft_cap_pct` con ventana `soft_cap_grace_sec` → `allow`/`soft_allow`/`block`
- **Auto‑close** (opcional): calcula `compute_auto_close_qty` para volver a límites

### 6.3 DailyGuard

- **Halt** si: pérdida diaria > `daily_max_loss_usdt`, drawdown > `daily_max_drawdown_pct` o racha de pérdidas > `max_consecutive_losses`
- Acción: `halt`/`close_all` (cierre forzado)

### 6.4 OCO (stop‑loss / take‑profit)

- `sl_pct`, `tp_pct`, `trail_pct`, `tp_tiers=[(R_multiple, fraction)]`
- **Trailing**: actualiza `sl_price` hacia el mejor precio sin relajarlo
- **Tiers**: cierre parcial por múltiplos de `R = |entry − sl|`

### 6.5 PnL y posiciones

- `apply_fill(pos, side, qty, px, fee_bps, venue_type)` → actualiza `avg_price`, `qty`, `realized_pnl`, `fees_paid`
- `mark_to_market(pos, mark_px)` → `UPnL` intradía

---

## 7) Ejecución: router, algoritmos y validación

### 7.1 Router multi‑exchange

- Selección por **(spread efectivo − fees − profundidad ponderada − latencia)**
- Trazabilidad en logs: venue elegido y factores

### 7.2 Algoritmos

- **TWAP(order, horizon\_s)**, **VWAP(order, participation)**, **POV(order, pct)**
- Fragmentación/ventanas y `reduce‑only` en cierres parciales (futuros)

### 7.3 Tipos de orden y flags

- **post‑only**, **IOC**, **FOK**, **reduce‑only**, `timeInForce`, iceberg (si disponible)

### 7.4 Slippage model

- Por **participación de volumen** y por **posición en cola** (microestructura)

### 7.5 Reglas del exchange

- Validar **stepSize/minNotional** con `adjust_order(meta.rules[symbol])`
- Abort si tamaño ajustado < mínimos

---

## 8) Persistencia de datos (TSDB)

### 8.1 Esquema TimescaleDB (resumen)

- `trades(ts, symbol, px, qty, side, id)`
- `orderbook_l2(ts, symbol, bid_px[], bid_qty[], ask_px[], ask_qty[])` (o forma long con nivel)
- `bars_1m(ts, symbol, o,h,l,c,v)`
- `funding(ts, symbol, rate, interval)`
- `oi(ts, symbol, value)`
- `orders(id, ts, symbol, side, qty, px, type, tif, status, venue)`
- `fills(id, order_id, ts, qty, px, fee_usdt, venue)`
- `positions(symbol, qty, avg_px, realized_pnl, fees_paid)`
- `pnl_snapshots(ts, symbol, realized, unrealized)`
- `risk_events(ts, type, payload)`

### 8.2 Writer asíncrono

- Bulk inserts con colas; **backpressure** y reintento

### 8.3 Backfill jobs

- `backfill(venue, symbols, days)` para OHLCV/trades y snapshots L2

---

## 9) Backtesting serio

### 9.1 Motores

- **vectorbt** para exploración masiva de parámetros y universos
- **Event‑driven** para simulaciones realistas (latencia, colas, funding, fees)

### 9.2 Métricas y reporting

- `SR, Sortino, Calmar, hit rate, max DD, slippage, maker/taker split, expectancy`
- **DSR (Deflated Sharpe Ratio)** y **walk‑forward** (purged k‑fold/embargo)
- Reportes `.html/.csv` automáticos

---

## 10) Monitoreo y panel

- **Exporters Prometheus**: `ws_uptime`, `ingest_rate`, `route_latency`, `order_reject_rate`, `pnl_realized`, `pnl_unrealized`
- **Panel FastAPI**: endpoints REST y vista web para: salud WS, posiciones/órdenes activas, PnL intradía, alertas y logs
- **Alertas**: faltas de ticks (>30s), DD intradía, rechazos de órdenes

---

## 11) CLI y Runners

**Comandos estándar** (Typer/Click):

- `ingest` / `ingest-historical` / `ingestion-workers`
- `backtest` / `backtest-cfg` / `walk-forward`
- `paper-run` / `testnet-run` / `real-run`
- `tri-arb` / `cross-arb` / `run-cross-arb`
- `report` / `daemon` / `run-bot`

**Runners de referencia**

- **Spot single/multi testnet**: WS → bars(1m) → BreakoutATR → balance‑cap → orden → Risk/PnL/OCO → persistencia
- **Futuros single/multi testnet**: idem + `reduce‑only`, apalancamiento y short/long simétrico
- **Rehidratación**: carga posiciones y OCOs activos al iniciar si `persist_pg` activo

---

## 12) Configuración (YAML)

```yaml
mode: paper  # paper|testnet|real
exchanges:
  - name: binance
    kind: spot
    testnet: true
symbols: ["BTCUSDT", "ETHUSDT"]
risk:
  risk_budget_usd: 1000
  daily_max_loss_usdt: 150
  daily_max_drawdown_pct: 3.0
  max_consecutive_losses: 4
  per_symbol_cap_usdt: 500
  total_cap_usdt: 1500
  soft_cap_pct: 0.15
  soft_cap_grace_sec: 300
  auto_close: true
  correlation_clusters:
    - ["BTCUSDT", "ETHUSDT"]
execution:
  algo: twap
  post_only: true
  ioc: false
  fok: false
  oco:
    enabled: true
    sl_pct: 0.006
    tp_pct: 0.012
    trail_pct: 0.004
    tp_tiers: [[1.0, 0.5], [2.0, 0.5]]
db:
  uri: postgresql://user:pass@localhost:5432/tradebot
monitoring:
  prometheus_port: 9108
  panel_port: 8000
```

---

## 13) Blueprint ↔ Código (mapa)

| Punto del blueprint | Módulos / archivos                                                                     | Comandos CLI                                                    |
| ------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| **1. Ingesta**      | `adapters/`, `data/ingestion.py`, `workers/`                                           | `ingest`, `ingest-historical`, `ingestion-workers`              |
| **2. Features**     | `data/features.py`, `data/basis.py`, `data/open_interest.py`                           | –                                                               |
| **3. Estrategias**  | `strategies/`, `live/runner*.py`                                                       | `run-bot`, `paper-run`, `tri-arb`, `cross-arb`, `run-cross-arb` |
| **4. Riesgo**       | `risk/manager.py`, `risk/portfolio_guard.py`, `risk/daily_guard.py`                    | `run-bot`, `daemon`                                             |
| **5. Ejecución**    | `execution/router.py`, `execution/algos.py`, `execution/order_types.py`                | `run-bot --algo`, `run-cross-arb`                               |
| **6. Persistencia** | `storage/timescale.py`, `storage/quest.py`                                             | `ingest*`, `report`                                             |
| **7. Backtesting**  | `backtest/event_engine.py`, `backtesting/engine.py`, `backtesting/vectorbt_wrapper.py` | `backtest*`, `walk-forward`, `paper-run`                        |
| **8. Monitoreo**    | `monitoring/panel.py`, `monitoring/metrics.py`, `utils/metrics.py`                     | `report`                                                        |

---

## 14) Pruebas (pytest)

**Unit tests**

- Features: OFI, imbalance, ATR/Donchian/Keltner (fixtures deterministas)
- Riesgo: sizing por ATR, hard/soft caps, auto‑close, daily guard, OCO (tiers/trailing)

**Integración**

- Ejecución (testnet): post‑only/IOC/FOK/OCO, cancelaciones, reintentos
- Backtest determinista: datasets pequeños en `data/samples/` (SR/PnL tol ±0.5 %)

---

## 15) Operación local

- **Docker Compose**: `db (Timescale)`, `grafana`, `prometheus`, `panel`
- **.env**: claves de exchange (sin retiros), flags testnet, URIs DB, puertos Prometheus/Panel
- **NTP** activo; hardware ≥ 8 cores, 32–64 GB RAM, SSD NVMe; disco 1–2 TB si L2 crudo

---

## 16) Seguridad y cumplimiento

- Claves en `.env` / secret manager local; **no** commitear
- Desactivar **retiros** en claves API; 2FA
- Respetar rate limits de cada venue; cumplir KYC/AML y normativa local

---

## 17) Extensibilidad

- **Nuevos exchanges**: implementar `*Adapter` REST/WS y mapear reglas (filters, fees)
- **Nuevas estrategias**: clase con interfaz `fit/transform/signal` o `signal(bar_window)` y registro en CLI
- **Nuevos features**: módulo en `data/features.py` con tests y docstring ejemplos
- **Nuevos algoritmos de ejecución**: función en `execution/algos.py` con simulación en backtester

---

## 18) Troubleshooting & hardening (pendientes típicos)

- **Etiquetas de logs**: evitar mensajes confusos (p. ej., "(spot)" en runners de futuros). Buscar y reemplazar.
- **Separar excepciones**: distinguir fallos al **enviar orden** vs **persistencia** para depurar correctamente.
- **Validación de mínimos**: siempre pasar por `adjust_order` y abortar si no cumple `stepSize/minNotional`.
- **Rehidratación**: al iniciar multi‑símbolo, cargar posiciones y OCOs activos y sincronizar `RiskManager.set_position`.
- **Fees configurables**: `taker_fee_bps` por adapter/entorno; registrar en PnL y snapshots.
- **Duplicación de lógica**: factorizar caps/OCO/daily guard/persistencia común a spot/futuros.

---

## 19) Glosario y variables de entorno

**ENV (ejemplo)**

- `DB_URI`, `PROM_PORT`, `PANEL_PORT`
- `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `BINANCE_TESTNET=1`
- `BYBIT_API_KEY`, `BYBIT_API_SECRET`, `BYBIT_TESTNET=1`
- `OKX_API_KEY`, `OKX_API_SECRET`, `OKX_PASSPHRASE`, `OKX_TESTNET=1`
- `TAKER_FEE_BPS_SPOT`, `TAKER_FEE_BPS_FUTURES`

**Siglas**

- **OFI**: Order Flow Imbalance; **BBA**: Best Bid/Ask; **OI**: Open Interest; **DSR**: Deflated Sharpe Ratio

---

### Anexo A) Ejemplo de CLI

```
# Ingesta tiempo real (Binance Spot)
python -m src.tradingbot.cli ingest --venue binance --symbols BTCUSDT,ETHUSDT --persist

# Backfill histórico 7 días
python -m src.tradingbot.cli ingest-historical --venue binance --symbols BTCUSDT --days 7

# Backtest estrategia Breakout ATR
python -m src.tradingbot.cli backtest --strategy breakout_atr --config config/default.yaml

# Paper-run 30 min
python -m src.tradingbot.cli paper-run --strategy breakout_atr --symbols BTCUSDT --minutes 30

# Testnet multi‑símbolo (futuros)
python -m src.tradingbot.cli testnet-run --venue binance_futures --symbols BTCUSDT,ETHUSDT --algo twap --oco
```

### Anexo B) Esquema SQL (borrador Timescale)

```sql
CREATE TABLE trades(
  ts TIMESTAMPTZ NOT NULL,
  symbol TEXT NOT NULL,
  px NUMERIC NOT NULL,
  qty NUMERIC NOT NULL,
  side SMALLINT NOT NULL,
  id TEXT
);
SELECT create_hypertable('trades','ts');
-- (Definir índices por ts/symbol; tablas análogas para el resto.)
```

---

> **Notas**: Este documento integra el blueprint conceptual, el mapeo blueprint→código y la definición técnica de componentes (runners, riesgo, OCO, PnL, ejecución, persistencia). Sirve como fuente única para desarrollo, pruebas y endurecimiento del MVP.

