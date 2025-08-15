# Blueprint Bot de Trading – Intradía, Scalping y Arbitraje

## 1) Objetivo y alcance

* **Alcance inicial (MVP)**: spot y perp-futuros en 2–3 exchanges (p. ej., Binance, Bybit, OKX), con módulos de: data en tiempo real (L1/L2), backtesting, ejecución de órdenes, gestión de riesgo, monitoreo y paper-trading. ([developers.binance.com][1], [bybit-exchange.github.io][2], [OKX][3])
* **Estrategias**: momentum intradía, mean reversion, microestructura (order flow/imbalance), breakout de volatilidad; arbitraje triangular (intra-exchange) y **cross-exchange**; cash-and-carry/funding-basis con perps. Funding y basis salen de endpoints nativos. ([econpapers.repec.org][4], [arXiv][5], [developers.binance.com][6])
* **No objetivos del MVP**: co-location, HFT extremo (<1 ms), opciones complejas.

---

## 2) Arquitectura técnica (modular, “plug‑and‑play”)

**Capas y módulos**

1. **Ingesta de datos**

   * WebSockets L1/L2 (precio, trades, order book) + REST para históricos.
   * Normalización por “adapter” de exchange (usamos **CCXT** para REST y un conector WS propio o CCXT Pro equivalente). ([docs.ccxt.com][7])

2. **Feature engineering**

   * Señales técnicas (retornos, RSI, ATR), microestructura (order‑flow imbalance, depth imbalance), eventos (funding/basis).

3. **Señal / estrategia**

   * Reglas determinísticas y/o ML (árboles/boosting/series temporales).

4. **Gestión de cartera y riesgo**

   * Tamaño de posición (volatilidad objetivo), límites de pérdida, kill‑switch, control de correlación entre símbolos.

5. **Ejecución**

   * Enrutador de órdenes multi‑exchange; algoritmos TWAP/VWAP/POV, **maker vs taker** con comisiones/rebates; control de slippage y colas.
   * Órdenes: limit/post-only/IOC/FOK, con “iceberg” si el exchange lo soporta.

6. **Persistencia**

   * Tick/Bar a **TimescaleDB** (Postgres) o **QuestDB** (si priorizamos ingestión bruta). Para local es sencillo desplegar ambos con Docker. ([GitHub][8], [Supabase][9], [QuestDB][10])

7. **Backtesting & simulación**

   * Motor event‑driven con latencia/colas simuladas, fees y reglas por exchange.

8. **Monitoreo & Ops**

   * Métricas Prometheus/Grafana, logging estructurado, alertas, panel mínimo (FastAPI + un dashboard ligero).

**Diagrama (alto nivel)**

```
[Exchanges WS/REST] -> [Adapters] -> [Bus interno asyncio]
                                \-> [Persistencia (Timescale/Quest)]
 [Feature Eng] -> [Estrategias] -> [Gestor Riesgo] -> [Router Ejecución] -> [Exchanges]
                                   [Backtester]  [Paper Trading]
                               [Monitoreo + Dash]   [Logs + Métricas]
```

---

## 3) Tecnologías y herramientas (Python-first)

* **Lenguaje**: Python 3.11+ (asyncio, uvloop), pydantic, numpy/pandas/numba, polars (opcional), Typer/Click para CLI.
* **Conectores**: **CCXT** (REST unificado multi‑exchange); WS nativo por exchange para L2 (Binance/Bybit/OKX). ([docs.ccxt.com][7], [developers.binance.com][1], [bybit-exchange.github.io][2], [OKX][3])
* **Backtest**:

  * Vectorizado masivo: **vectorbt** (rápido para explorar universos/parametría). ([vectorbt.dev][11])
  * Event‑driven propio (para slippage microestructura) + **Freqtrade** como baseline crypto (tiene backtest/optimización integrados). ([freqtrade.io][12])
* **Datos históricos/alternativos** (si no queremos mantener pipelines crudos): **Kaiko**, **CoinAPI** (L1/L2, trades, OI, funding, snapshots). Útiles para enriquecer y validar. ([kaiko.com][13], [coinapi.io][14])
* **TSDB**: **TimescaleDB** (facilidad Postgres) vs **QuestDB** (ingestión ultra‑rápida para ticks). ([GitHub][8], [QuestDB][10])
* **Experimentos**: MLflow (local), Optuna (búsqueda bayesiana), Hydra (configs).
* **Monitoreo**: Prometheus/Grafana; Sentry para errores.
* **Pruebas**: pytest, hypothesis (tests de propiedades), **fixtures** de mercado.

---

## 4) Exchanges objetivo y entornos de prueba

* **Binance (USDⓈ‑M/COIN‑M Futures y Spot)**: endpoints REST/WS, **testnet** para futures (paper trading). Funding endpoints públicos para capturar basis/funding. ([developers.binance.com][1], [binance-docs.github.io][15])
* **Bybit**: API v5 unificada, **testnet** claro (`api-testnet.bybit.com`). ([bybit-exchange.github.io][2])
* **OKX**: REST/WS v5; docs completas. (Testnets y productos spot/derivados). ([OKX][3])
* **Opcionales (derivados avanzados)**: Deribit (perps/opciones; funding especificado y endpoints de interés). ([Deribit Soporte][16], [Deribit API][17])

---

## 5) Datos que necesitamos (tiempo real & histórico)

* **Tiempo real**:

  * **Trades** (tick‑by‑tick), **Order Book L2** (profundidad n>10), **Best Bid/Ask**, **OI** y **funding rate** en perps. ([developers.binance.com][6])
* **Histórico**:

  * OHLCV multi‑resolución (1s–1h), trades y snapshots de libro para backtests micro‑estructurales.
  * **Fuentes**: Exchange directos (REST), **Kaiko** / **CoinAPI** si queremos descargar grandes repositorios ya normalizados. ([kaiko.com][13], [coinapi.io][14])

---

## 6) Señales y estrategias (lo que sí funciona intradía)

**A) Microestructura (L2 / flujo de órdenes)**

* **Order Flow Imbalance (OFI)**: relación lineal con cambios de precio a corto plazo (señal clásica). Implementar OFI en mejor bid/ask y probar extensiones multi‑nivel. ([econpapers.repec.org][4], [ORA][18])
* **Queue/Depth Imbalance**: probabilidad del próximo tick (clasificación). ([arXiv][5])
* **Eventos de liquidez**: vaciados de book, gaps de profundidad, barridos de stop.

**B) Momentum intradía y “breakouts”**

* Retornos acumulados 1–5–15m, Keltner/Donchian, volatilidad adaptativa (ATR‑EWMA).
* Confirmación con **trade imbalance** (agresiones netas).

**C) Mean Reversion rápida**

* Reversiones tras extremos de OFI o spikes de volatilidad; filtros de spread/latencia.

**D) Arbitrajes**

* **Triangular (intra‑exchange)**: detectar rutas A/B/C con edge neto (tras fees/slippage).
* **Cross‑exchange**: misma paridad en 2+ exchanges (latencias, balances, límites de retiro).
* **Cash‑and‑carry / Funding basis**: largos spot vs cortos perp cuando funding positivo (o inverso), capturando carry ajustado por fees. Fórmulas/limitaciones en docs de exchanges. ([Deribit Insights][19], [Deribit Soporte][16])

**E) ML opcional (no en el día 1)**

* Etiquetado **Triple Barrier** (+ meta‑labeling) para reducir ruido en labels. ([williamsantos.me][20])
* Validación robusta con **DSR (Deflated Sharpe Ratio)** para evitar “overfitting” de backtests. ([SSRN][21])

---

## 7) Modelado de costos, latencia y ejecución

* **Comisiones maker/taker** por exchange y nivel VIP; rebates maker pueden convertir señales marginales en rentables.
* **Slippage**:

  * Modelo por **participación de volumen** y por “queue position” en L2.
  * Impacto por OFI (microestructura) adaptado de literatura académica. ([econpapers.repec.org][4])
* **Algoritmos de ejecución**:

  * **Post‑only** (evitar taker), **IOC/FOK** (escapar de libros finos), **TWAP/VWAP/POV** para liberar inventario.
* **Routing**: escoger venue por (spread efectivo – fees – profundidad – latencia).

---

## 8) Gestión de riesgo (runtime y por backtest)

* **Sizing**: volatilidad objetivo (p. ej., 10–20% anualizado por estrategia), cap de apalancamiento.
* **Stops**: por barra/tiempo/volatilidad; hard‑stop de PnL diario y **kill‑switch** global.
* **Drawdown rules**: escalado dinámico (de‑risk) al exceder X% DD.
* **Agregación de riesgo**: correlaciones intradía entre símbolos e intercambios.
* **Validación estadística**: SR/Sortino/Calmar; **DSR** para corregir sesgo por múltiples pruebas; “walk‑forward” con purged k‑fold/embargo. ([SSRN][21])

---

## 9) Backtesting serio (evitar sesgos)

* **Motores**:

  * Exploración rápida con **vectorbt** (miles de combinaciones en segundos). ([vectorbt.dev][11])
  * Event‑driven propio para: latencia, colas L2, fills parciales, funding, fees.
  * **Freqtrade** como referencia crypto (backtesting/optimización integrados y estrategias de ejemplo). ([freqtrade.io][12], [GitHub][22])
* **Datos**: preferir **trades+L2** y reconstruir OHLCV; cuidado con survivorship bias y relojes.
* **Slippage & Fees**: obligatorios y realistas (tier actual).
* **Robustez**: test de estrés (latencia x2, spreads x2, caída de liquidez).
* **Evaluación**: PnL intradía, “edge per trade”, tasa de fill, impacto de latencia, consumo de margen/funding, **deflated SR**. ([SSRN][21])

---

## 10) Persistencia y esquemas de datos

* **TimescaleDB / QuestDB** tablas separadas por `exchange/symbol/type`

  * `trades(ts, px, qty, side, id)`
  * `orderbook(ts, bid_px[], bid_qty[], ask_px[], ask_qty[])` (o forma “long” con nivel)
  * `bars(ts, o,h,l,c,v)`
  * `funding(ts, symbol, rate, interval)` (perps) ([GitHub][8], [QuestDB][10])

---

## 11) Seguridad, claves y cumplimiento

* **Claves**: .env + “secret manager” local (por ahora), sin imprimir ni subir a git.
* **Permisos**: **desactivar retiros** en claves API; 2FA.
* **Rate limits**: respetar límites de cada exchange para evitar bans. ([developers.binance.com][1])
* **KYC/AML y fiscal**: operar conforme a términos de cada exchange y normatividad local (revisar con tu asesor).

---

## 12) Operación en máquina local (setup recomendado)

* **Hardware**: CPU moderna (≥8 cores), 32–64 GB RAM, SSD NVMe; si guardamos L2 crudo, disco ≥1–2 TB.
* **Sistema**: Docker para Timescale/Quest, Redis opcional (colas), FastAPI para API interna, servicio único de orquestación basado en `asyncio` (no necesitamos Kafka en el MVP).
* **Time sync**: NTP activo (relojes exactos para backtests/fills).
* **Dash**: panel mínimo con métricas clave y control de estrategias.

---

## 13) Métricas de desempeño y salud

* **Trading**: PnL neto, SR/Sortino, hit rate, expectancy, duración media trade, **slippage real vs estimado**, ratio maker/taker, % órdenes rechazadas/canceladas.
* **Mercado**: spreads, profundidad, OFI, volatilidad instantánea.
* **Sistema**: latencia E2E (tick → orden), CPU/ram, caídas de WS, re‑suscripciones.

---

## 14) Roadmap por fases (90 días)

**Fase 0 (Semana 1–2) – Cimientos**

* Repos monorepo, entorno, linters, configs Hydra, logging, kits de tests.
* Conectores REST/WS de 2 exchanges + almacenamiento Timescale/Quest. ([GitHub][8], [QuestDB][10])

**Fase 1 (Semanas 3–5) – Datos & Backtest**

* Ingesta L1/L2 en tiempo real y descarga histórica.
* Backtester vectorizado (vectorbt) y primer motor event‑driven. ([vectorbt.dev][11])

**Fase 2 (Semanas 6–8) – Estrategias base**

* Momentum/Breakout/Mean‑reversion; OFI/imbalance; arbitraje triangular.
* Simulación de ejecución (maker/post‑only vs taker).

**Fase 3 (Semanas 9–10) – Ejecución real & riesgo**

* Router de órdenes, límites, kill‑switch, paper‑trading/testnet; funding‑carry simple. ([binance-docs.github.io][15])

**Fase 4 (Semanas 11–13) – Validación & endurecimiento**

* Walk‑forward, estrés, **DSR**; panel de monitoreo y alertas. ([SSRN][21])

---

## 15) Entregables del MVP

* **Daemon** de trading (FastAPI/CLI) con módulos: data, señales, riesgo, ejecución.
* **Backtester** + reportes automáticos (métricas, gráficos).
* **TSDB** con esquemas definidos y scripts de carga.
* **Panel** de estado y métricas en tiempo real.
* **Documentación** para empaquetar y escalar (y base para una UI comercializable).

---

## 16) Decisiones por omisión (modificables)

* Lenguaje: **Python** (rapidez de I+D; Numba/Polars si hace falta).
* TSDB: **TimescaleDB** por compatibilidad Postgres (QuestDB como opción si la ingestión es cuello de botella). ([GitHub][8], [QuestDB][10])
* Exchanges: **Binance + Bybit** (liquidez, buen testnet), **OKX** (alternativa). ([developers.binance.com][1], [bybit-exchange.github.io][2], [OKX][3])
* Backtesting inicial: **vectorbt** + motor propio para microestructura. ([vectorbt.dev][11])
