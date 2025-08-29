# TradeBot

TradeBot es un bot de trading educativo y de código abierto para mercados de
criptomonedas. El proyecto incluye herramientas para descargar datos,
realizar backtesting, ejecutar estrategias en modo prueba o con dinero real
y monitorear el desempeño con tableros de control.

La documentación completa está en la carpeta [`docs`](docs/). Allí se
explican los conceptos básicos, las estrategias disponibles y todas las
opciones de la línea de comandos.

## Requisitos mínimos

- Python 3.11
- Dependencias listadas en `requirements.txt`

## Instalación rápida

```bash
pip install -e .
```

## Uso básico

Todos los comandos se ejecutan con `python -m tradingbot.cli` seguido del
subcomando deseado. Por ejemplo, para ver la ayuda general:

```bash
python -m tradingbot.cli --help
```

Para una guía detallada de cada comando, consulte [docs/commands.md](docs/commands.md).

## Modo testnet en el CLI

Para operar contra los entornos de prueba de los exchanges se puede
añadir la opción `--testnet` a cualquier comando del CLI. Por ejemplo:

```bash
python -m tradingbot.cli run --testnet
```

Antes de ejecutar en este modo es necesario definir las credenciales de
cada exchange mediante variables de entorno:

- `BINANCE_TESTNET_API_KEY` y `BINANCE_TESTNET_API_SECRET`
- `BYBIT_TESTNET_API_KEY` y `BYBIT_TESTNET_API_SECRET`
- `OKX_TESTNET_API_KEY`, `OKX_TESTNET_API_SECRET` y `OKX_TESTNET_API_PASSPHRASE`

Puedes copiar estas claves desde `.env.example` a tu archivo `.env` o
exportarlas en la terminal antes de ejecutar el bot.

## Variables de entorno

De forma predeterminada, Binance Futures opera contra el entorno real. Para
volver a utilizar la testnet es necesario indicar explícitamente:

```bash
export BINANCE_FUTURES_TESTNET=true
```

Si esta variable no está definida o se establece en `false`, el bot usará el
entorno real de Binance Futures.

## Gestión de riesgo

Las señales incluyen un parámetro `strength` que dimensiona la orden según
`notional = equity * strength`. Un `strength = 1.0` usa todo el capital
disponible; valores mayores piramidan la posición y menores la reducen. El
gestor de riesgo limita la pérdida con `risk_pct` y permite dimensionar según
la volatilidad mediante `vol_target`. Para limitar la exposición global puede
emplearse `PortfolioGuard`, dejando `total_cap_pct` y `per_symbol_cap_pct` en
`null` para deshabilitar estos límites.

El parámetro `risk_pct` debe indicarse como una fracción entre 0 y 1. Si se
provee un valor de 1 a 100 se interpreta como porcentaje y se divide entre
100 (por ejemplo, `risk_pct=1` equivale a `0.01`). Valores negativos o
superiores a 100 generan un error.

Ejemplos:

```bash
python -m tradingbot.cli backtest data.csv --symbol BTC/USDT --risk-pct 0.02
# equivalente a:
python -m tradingbot.cli backtest data.csv --symbol BTC/USDT --risk-pct 2
# y "--risk-pct 1" equivale a 0.01 (1 % de riesgo)
```

`DailyGuard` supervisa las pérdidas intradía y el drawdown global. Si se
superan los límites configurados, detiene el bot o cierra las posiciones
abiertas.

### Ejemplo para cuenta pequeña

```yaml
backtest:
  data: data/examples/btcusdt_1m.csv   # reemplazar con datos del activo elegido
  symbol: DOGE/USDT
  strategy: breakout_atr
  initial_equity: 100

risk:
  risk_pct: 0.02
  vol_target: 0.01
  total_cap_pct: null
  per_symbol_cap_pct: null

```

El motor de backtesting ignora ejecuciones cuya cantidad sea menor a
`min_fill_qty` para evitar registrar residuos irrelevantes. El umbral
predeterminado es la constante `MIN_FILL_QTY = 1e-3`, pero puede ajustarse
al crear `EventDrivenBacktestEngine` o mediante los campos `backtest.min_fill_qty`
y `exchange_configs.<exchange>.min_fill_qty` en la configuración YAML.

## Solución de problemas

Si se muestra el mensaje `System clock offset`, indica que el reloj del
sistema está desincronizado. Se recomienda sincronizar la hora con un servidor
NTP para evitar inconsistencias o, si la verificación no es crítica, desactivar
la comprobación ajustando la variable de entorno `NTP_OFFSET_THRESHOLD` a un
valor alto.

## Recursos

- [Blueprint de arquitectura](BLUEPRINT.md)
- [MVP: características mínimas](MVP.md)
- [Glosario de términos](docs/glossary.md)
- [Estrategias disponibles](docs/strategies.md)
- [Dashboards de monitoreo](docs/dashboards.md)

## Licencia

Este proyecto se distribuye bajo la licencia MIT.
