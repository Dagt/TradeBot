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

`RiskService` centraliza la gestión combinando el `CoreRiskManager`,
`PortfolioGuard` y `DailyGuard`. Se inicializa junto a una
[`Account`](src/tradingbot/core/account.py) que lleva el registro de la
exposición. Las estrategias emiten señales con un campo `strength` continuo
(`notional = equity * strength`) y el servicio dimensiona la orden y aplica un
trailing stop adaptativo basado en la volatilidad (ATR).

El riesgo por operación se controla mediante `risk_pct` (equivalente a
`--risk-pct` en la CLI). Para limitar la exposición global pueden utilizarse
los parámetros `total_cap_pct` y `per_symbol_cap_pct` de `PortfolioGuard`.

### Ejemplo

```python
from tradingbot.core.account import Account
from tradingbot.risk.portfolio_guard import GuardConfig, PortfolioGuard
from tradingbot.risk.service import RiskService

account = Account(cash=1000.0)
guard = PortfolioGuard(GuardConfig(venue="demo"))
risk = RiskService(guard, account=account, risk_pct=0.02)  # --risk-pct 2 en la CLI

signal = {"side": "buy", "strength": 0.5}
price = 100
allowed, _reason, delta = risk.check_order(
    "BTC/USDT", signal["side"], price, strength=signal["strength"]
)
if allowed and delta > 0:
    # enviar orden usando delta como cantidad
    pass
```

El parámetro `risk_pct` debe indicarse como una fracción entre 0 y 1.
Si se provee un valor de 1 a 100 se interpreta como porcentaje y se divide
entre 100 (por ejemplo, `--risk-pct 1` equivale a `0.01`). Valores negativos
o superiores a 100 generan un error.

`DailyGuard` supervisa las pérdidas intradía y el drawdown global. Si se
superan los límites configurados, detiene el bot o cierra las posiciones
abiertas.

### Ejemplo para cuenta pequeña

Las configuraciones de ejemplo utilizan velas de 3 minutos.

```yaml
backtest:
  data: data/examples/btcusdt_3m.csv   # reemplazar con datos del activo elegido
  symbol: DOGE/USDT
  strategy: breakout_atr
  initial_equity: 100

risk:
  risk_pct: 0.02
  total_cap_pct: null
  per_symbol_cap_pct: null

```

El modelo de *slippage* admite dos fuentes (`source`):

- `"bba"` utiliza las columnas `bid`/`ask` (o `bid_px`/`ask_px`) si están
  presentes y recurre al `base_spread` configurado cuando faltan o contienen
  valores `NaN`.
- `"fixed_spread"` siempre aplica el valor de `base_spread` sin considerar las
  columnas de mejor bid/ask.

Ejemplo en la configuración:

```yaml
backtest:
  slippage:
    source: bba
    base_spread: 0.1  # spread fijo si faltan columnas bid/ask o tienen NaN
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
