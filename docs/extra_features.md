# Funcionalidades extra de TradeBot

Este documento resume capacidades que van más allá del MVP inicial.
Cada una se describe con un detalle técnico breve y un ejemplo de uso
para que también sea comprensible para quien no tiene experiencia previa.

## Estrategias avanzadas

Además de las estrategias básicas del MVP (momentum, mean reversion y
breakout), el bot incluye módulos listos para:

- **Arbitraje triangular**: detecta desalineaciones entre tres pares en un
  mismo exchange y ejecuta las tres patas en una sola operación.
  ```bash
  python -m tradingbot.cli tri-arb BTC-ETH-USDT --notional 100
  ```
- **Arbitraje entre exchanges / cash-and-carry**: compara precios spot vs
  futuros o entre venues y abre posiciones opuestas.
  ```bash
  python -m tradingbot.cli cross-arb BTC/USDT binance_spot binance_futures \
      --threshold 0.001 --notional 50
  ```
- **Microestructura**: señales basadas en Order Flow Imbalance (OFI),
  profundidad de libro y desequilibrio de colas para scalping.

Estas estrategias pueden ejecutarse en modo ``paper`` o en cuentas reales.
Para habilitar trading en producción utiliza el comando `real-run` y
confirma con la opción `--i-know-what-im-doing`.

## Adaptadores y conectores

El proyecto expone adaptadores REST/WebSocket para **Binance**, **Bybit**
 y **OKX**, con soportes tanto para spot como para futuros y sus testnets.
La estructura es modular para facilitar la inclusión de nuevos exchanges.

Ejemplo rápido para obtener trades de Bybit:
```bash
python -m tradingbot.cli ingest --venue bybit_spot --symbol BTC/USDT
```

Para completar históricos existe el comando **backfill**, que descarga
OHLCV y trades en bloque respetando los límites de cada API:

```bash
python -m tradingbot.cli backfill --days 30 --symbols BTC/USDT ETH/USDT
```

## Monitoreo y panel web

La aplicación FastAPI incluye un **dashboard** en `/` con métricas en
 tiempo real, PnL, exposición y eventos de riesgo.  También posee un
 ejecutor de comandos para lanzar cualquier instrucción de la CLI desde el
 navegador, permitiendo la gestión sin necesidad de terminal.  El panel
 ofrece formularios en modo oscuro donde se pueden configurar exchanges,
 claves API y parámetros de estrategias de forma interactiva.

Ejemplos:

- Ejecutar un backtest desde la interfaz web escribiendo en la caja
  `backtest data/btc.csv --strategy breakout_atr`.
- Definir las credenciales de Binance y el umbral de una estrategia
  `momentum` sin usar la línea de comandos.


## Backtesting y análisis

- Motor **vectorizado** basado en `vectorbt` para exploración rápida (instalar con `pip install "vectorbt>=0.26"`).
- Motor **event-driven** que simula profundidad L2, slippage y latencia.
- Utilidades de **walk-forward** y generación de reportes con métricas
  como Sharpe, Sortino y drawdown.

## Gestión de riesgo ampliada

- **PortfolioGuard** con límites duros y blandos por símbolo y globales.
- **DailyGuard** que corta operaciones al exceder pérdidas diarias.
- **CorrelationService** para limitar exposición en activos altamente
  correlacionados.

## API y control remoto

El módulo `apps.api` expone endpoints protegidos por autenticación básica
para consultar órdenes, señales, PnL o pausar el bot.  Ejemplo con `curl`:
```bash
curl -u admin:admin http://localhost:8000/risk/halt -X POST -d '{"reason":"manual"}'
```

## Suite de pruebas

Más de un centenar de pruebas unitarias e integrales verifican la
integridad de las estrategias, del router de ejecución y de los servicios
de riesgo y monitoreo.  Se ejecutan con:
```bash
pytest
```

Estas funcionalidades extra amplían considerablemente el alcance del MVP
original y convierten a TradeBot en una plataforma completa para
explorar, probar y desplegar estrategias de trading cuantitativo.

