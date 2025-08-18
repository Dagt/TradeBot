# TradeBot

TradeBot es un bot de trading modular orientado a estrategias intradía,
scalping y arbitraje sobre criptomonedas.  Incluye todo lo necesario para
ingerir datos, realizar backtesting y ejecutar estrategias en modo
``paper`` o real desde una interfaz web.

### Explicación para principiantes

Un bot de trading es como un "piloto automático" que compra y vende
criptomonedas siguiendo un conjunto de reglas.  Tú defines las reglas y el
programa las ejecuta de forma rápida y sin emociones.  TradeBot ya trae
estrategias listas para usar y permite probarlas sin arriesgar dinero
gracias al **paper trading** (simulación).

## Características principales

- Ingesta de datos en tiempo real (WebSocket y REST) para Binance, Bybit,
  OKX y otros exchanges.
- Motor de estrategias con ejemplos de momentum, mean reversion, arbitraje
  triangular y arbitraje spot/perp.
- Gestión de riesgo y portafolio con límites de exposición y ``kill switch``.
- Router de ejecución con algoritmos TWAP/VWAP/POV y soporte maker/taker.
- Backtester vectorizado y motor event‑driven con modelado de slippage.
- **Panel web** con métricas en vivo y un **ejecutor de comandos CLI** que
  permite lanzar cualquier comando desde el navegador.

## Exchanges y pares soportados

- **Exchanges**: [Binance](https://www.binance.com),
  [Bybit](https://www.bybit.com) y [OKX](https://www.okx.com).  El diseño es
  modular y pueden añadirse más.
- **Mercados**: pares spot y contratos perpetuos disponibles en esos
  exchanges.
- **Pares populares**: BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT y cualquier
  otro listado por los exchanges anteriores.

## Estrategias incluidas

Cada estrategia se puede ejecutar en modo simulación o real.  A
continuación se explica la idea teórica y cómo la implementa TradeBot.

### Momentum intradía
**Idea**: cuando un precio sube con fuerza, suele seguir subiendo a corto
plazo.
**Implementación**: el bot calcula el retorno de los últimos minutos y
compra si supera un umbral; vende cuando el impulso se agota o cambia de
signo.

### Mean reversion
**Idea**: los precios tienden a volver a su promedio después de moverse
demasiado.
**Implementación**: se calcula una media móvil y su desviación.  Si el
precio está muy por encima, se vende; si está por debajo, se compra.

### Breakout de volatilidad
**Idea**: tras un periodo de calma, un movimiento brusco puede iniciar una
nuevo recorrido de precios.
**Implementación**: se observa la volatilidad (ATR) y se activan órdenes
cuando el precio rompe un canal predefinido.

### Arbitraje triangular
**Idea**: en un mismo exchange, las tasas de cambio entre tres pares pueden
quedar desalineadas.  Al hacer la ruta A→B→C→A se obtiene una ganancia
sin exposición direccional.
**Implementación**: el bot revisa continuamente rutas como BTC‑ETH‑USDT y
ejecuta las tres operaciones si el beneficio neto supera las comisiones.

### Arbitraje entre exchanges / cash‑and‑carry
**Idea**: un mismo activo puede tener precios distintos entre exchanges o
entre mercado spot y perp.  Comprar donde está barato y vender donde está
caro permite capturar la diferencia o el pago de funding.
**Implementación**: el bot compara precios de los exchanges conectados y
abre posiciones opuestas (spot vs perp o exchange vs exchange) cuando la
brecha supera un umbral configurado.

## Requisitos

- Python 3.11+
- ``git``
- ``docker`` (opcional, para bases de datos y monitoreo)

## Instalación rápida

```bash
git clone <repo>
cd TradeBot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # completa con tus claves
```

## Ejemplos de uso

Estos ejemplos utilizan datos de prueba y son seguros para principiantes.

### Simular una estrategia de momentum

```bash
python -m tradingbot.cli backtest-cfg data/examples/backtest.yaml
```

El comando ejecuta la estrategia de momentum sobre el par BTC/USDT y muestra
estadísticas como ganancias/pérdidas y número de operaciones.

### Probar un arbitraje triangular

```bash
python -m tradingbot.cli tri-arb BTC-ETH-USDT --notional 50
```

El bot revisa si la ruta BTC→ETH→USDT→BTC en Binance ofrece un beneficio
mayor a las comisiones usando un capital ficticio de 50 USDT.

## Ejecución del panel web

El panel expone métricas, PnL y una consola para ejecutar comandos de la
CLI.

```bash
uvicorn tradingbot.apps.api.main:app --reload --port 8000
```

Visita `http://localhost:8000` e inicia sesión con las credenciales
definidas en `API_USER`/`API_PASS` (por defecto `admin`/`admin`).

### Consola de comandos

En la sección **Comandos CLI** del panel puedes ejecutar cualquier comando
de `tradingbot.cli`.  Ejemplos:

1. Escribe `backtest-cfg data/examples/backtest.yaml` y pulsa **Ejecutar**.
2. Usa `tri-arb BTC-ETH-USDT --notional 50` para disparar un arbitraje
   triangular de prueba.

La salida de `stdout` y `stderr` aparecerá debajo del formulario.

## Uso desde la línea de comandos

La CLI está basada en [Typer](https://typer.tiangolo.com/) y ofrece
subcomandos para las distintas tareas del proyecto.

```bash
python -m tradingbot.cli --help

# Ingesta de libro de órdenes
python -m tradingbot.cli ingest BTC/USDT --depth 20

# Descarga histórica desde Kaiko
python -m tradingbot.cli ingest-historical kaiko BTC/USDT --kind trades

# Backtest a partir de un YAML de configuración
python -m tradingbot.cli backtest-cfg data/examples/backtest.yaml

# Arbitraje triangular
python -m tradingbot.cli tri-arb BTC-ETH-USDT --notional 50
```

Todos estos comandos pueden ejecutarse también desde el panel web gracias a
la nueva sección de **Comandos CLI**.

## Ejecutar pruebas

El proyecto incluye una batería extensa de tests.  En entornos con recursos
limitados puede ejecutarse una versión reducida:

```bash
PYTHONPATH=src:. pytest tests/test_smoke.py
```

## Estructura del proyecto

```
TradeBot/
├─ src/tradingbot/          # Código fuente del bot
├─ monitoring/              # Métricas y paneles de observabilidad
├─ docs/                    # Documentación adicional
├─ tests/                   # Pruebas unitarias e integración
└─ data/, sql/, bin/, etc.  # Scripts y ejemplos
```

---

Este repositorio es una base extensible.  Añade tus propias estrategias y
conectores según sea necesario y utilízalo tanto para uso personal como para
compartirlo con terceros.

