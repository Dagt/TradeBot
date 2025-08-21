# Guía de Usuario de TradeBot

Esta guía explica paso a paso cómo instalar, configurar y utilizar TradeBot.
Está escrita pensando en personas sin experiencia previa en Python, trading
algorítmico o bots.

## 1. ¿Qué es TradeBot?

TradeBot es un "piloto automático" para operar criptomonedas. Permite:

* Descargar precios reales de distintos exchanges.
* Probar estrategias con datos históricos para saber qué habría pasado en el
  pasado (**backtesting**).
* Ejecutar las estrategias en tiempo real sin arriesgar dinero (**paper
  trading**).
* Operar en exchanges reales cuando se tenga experiencia y claves de acceso.
* Monitorear resultados y métricas en un panel web.

## 2. Requisitos previos

1. **Git** para clonar el proyecto.
2. **Python 3.10 o superior**.
3. **pip** (se instala junto con Python) para descargar dependencias.
4. (Opcional) **Docker** si deseas utilizar las bases de datos y paneles
   incluidos.

## 3. Clonar e instalar

1. Abre una terminal.
2. Ejecuta:

```bash
git clone https://github.com/<tu_usuario>/TradeBot.git
cd TradeBot
```

3. Crea un entorno virtual para aislar las dependencias:

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

4. Instala las librerías necesarias:

```bash
pip install -r requirements.txt
```

5. (Opcional) Copia las variables de ejemplo y completa tus credenciales:

```bash
cp .env.example .env
```

En el archivo `.env` encontrarás espacios para poner tus claves de Binance,
Bybit u otros exchanges. Si sólo harás pruebas con datos históricos o en modo
simulación puedes dejarlos vacíos.

6. Levanta la base de datos y el panel de monitoreo con Docker:

```bash
./entrypoint.sh
```

Esto descargará los contenedores y los dejará listos. Más adelante podrás
acceder a un panel en <http://localhost:8000>.

## 4. Conceptos básicos

A continuación se definen los términos más usados. Cada sección incluye una
explicación sencilla y un ejemplo de uso.

### Backtest

Usa datos históricos reales (precios del pasado).

Sirve para responder: "¿qué habría pasado si hubiera corrido mi estrategia entre
enero y junio 2024?".

100% histórico → NO inventa datos, usa velas/trades reales descargados.

Ejemplo:

```bash
python -m tradingbot.cli backtest data/btcusdt_1m.csv --symbol BTC/USDT --strategy breakout_atr
```

### Backfill

Descarga datos faltantes de un exchange para completar huecos históricos.

Responde: "Necesito las últimas 48 horas de velas de BTC/USDT para poder probar
mi estrategia".

Ejemplo:

```bash
python -m tradingbot.cli backfill --days 2 --symbols BTC/USDT
```

### Ingest

Conecta al exchange y guarda en una base de datos el flujo de operaciones en
tiempo real. Estos datos luego se utilizan para estrategias o análisis.

Ejemplo:

```bash
python -m tradingbot.cli ingest --symbol BTC/USDT
```

### Paper Run (simulación en vivo)

Ejecuta la estrategia con precios en vivo pero **sin enviar órdenes reales**.
Sirve para practicar sin riesgo.

Ejemplo:

```bash
python -m tradingbot.cli paper-run --strategy breakout_atr --symbol BTC/USDT
```

### Run Bot / Real Run

`run-bot` ejecuta una estrategia en vivo o en testnet. `real-run` envía órdenes
al exchange real y requiere confirmar que se entiende el riesgo.

Ejemplos:

```bash
python -m tradingbot.cli run-bot --exchange binance --symbol BTC/USDT
python -m tradingbot.cli real-run --exchange binance --symbol BTC/USDT --i-know-what-im-doing
```

### Daemon

Inicia un servicio que gestiona múltiples estrategias basándose en un archivo de
configuración (usa [Hydra](https://hydra.cc)).

```bash
python -m tradingbot.cli daemon src/tradingbot/config/config.yaml
```

### Report

Genera un resumen de ganancias y pérdidas almacenadas en la base de datos.

```bash
python -m tradingbot.cli report --venue binance_spot_testnet
```

### Tri-Arb

Busca oportunidades de **arbitraje triangular** dentro de un mismo exchange.
Convierte de una moneda a otra y vuelve a la original intentando capturar una
ganancia en el ciclo.

```bash
python -m tradingbot.cli tri-arb BTC-ETH-USDT --notional 100
```

### Cross-Arb

Detecta diferencias de precio entre un mercado spot y uno de futuros
perpetuos. Si la diferencia supera un umbral se envían órdenes opuestas en
cada mercado.

```bash
python -m tradingbot.cli cross-arb BTC/USDT binance_spot binance_futures --threshold 0.001 --notional 50
```

### Descargar tasas de funding u open interest

El script `bin/download_history.py` puede obtener datos adicionales como
funding o interés abierto para análisis más avanzados.

```bash
python bin/download_history.py funding kaiko BTC-USD --exchange binance --backend csv
python bin/download_history.py open-interest coinapi BTCUSD --backend csv
```

### Backtest mediante archivo de configuración

Para organizar parámetros complejos puedes describir todo en un archivo YAML y
usarlo con `backtest-cfg`.

```bash
python -m tradingbot.cli backtest-cfg src/tradingbot/config/config.yaml
```

## 5. Configuración de estrategias

Las estrategias se seleccionan con el parámetro `--strategy` y aceptan
parámetros extra vía `--config` o a través de archivos YAML. Los nombres
disponibles son los que figuran en la columna `Nombre` de la tabla siguiente.

| Estrategia            | Nombre CLI            | Idea principal | Señal de compra | Señal de venta |
|-----------------------|----------------------|----------------|-----------------|----------------|
| Momentum              | `momentum`           | Seguir la tendencia usando RSI y flujo de órdenes. | RSI alto + presión compradora. | RSI bajo + presión vendedora. |
| Reversión a la media  | `mean_reversion`     | Comprar cuando el precio se aleja demasiado por abajo y vender si sube de más. | RSI < nivel inferior. | RSI > nivel superior. |
| Ruptura ATR           | `breakout_atr`       | Usa canales de Keltner (EMA+ATR). | Precio rompe por arriba. | Precio rompe por abajo. |
| Ruptura Volatilidad   | `breakout_vol`       | Media ± desviación estándar. | Cierre > media + k·desv. | Cierre < media - k·desv. |
| Flujo de Órdenes      | `order_flow`         | Evalúa el desequilibrio entre compras y ventas recientes. | OFI promedio > umbral. | OFI promedio < -umbral. |
| Reversión OFI         | `mean_rev_ofi`       | Vende cuando hay mucha presión compradora con baja volatilidad y compra lo contrario. | OFI z-score < -umbral. | OFI z-score > umbral. |
| Desequilibrio de Profundidad | `depth_imbalance` | Compara la cantidad de órdenes en bid vs ask. | Promedio > umbral. | Promedio < -umbral. |
| Eventos de Liquidez   | `liquidity_events`   | Detecta vacíos o gaps en el libro de órdenes. | Vacío del lado de ventas o gran gap hacia arriba. | Vacío del lado de compras o gran gap hacia abajo. |
| Cash & Carry          | `cash_and_carry`     | Explota diferencias entre spot y futuros perpetuos considerando el funding. | Funding positivo y basis alta. | Funding negativo y basis baja. |
| Arbitraje cruzado     | `cross_exchange_arbitrage` | Compara precio spot y perp entre exchanges distintos. | Premium > umbral, compra spot/vende perp. | Premium < -umbral, vende spot/compra perp. |
| Arbitraje triangular  | `triangular_arb`     | Recorre 3 mercados (A/B, B/C, C/A) buscando un ciclo rentable. | Edge a favor en ruta Base→Mid. | Edge a favor en ruta Mid→Base. |
| Arbitraje simple      | `arbitrage`          | Plantilla para extender. No genera órdenes por defecto. | — | — |
| Triple barrera (ML)   | `triple_barrier`     | Usa un modelo de aprendizaje automático con etiquetas de barrera superior/inferior. | Modelo predice subida. | Modelo predice caída. |
| Estrategia ML genérica| `ml`                 | Carga un modelo de scikit-learn entrenado con tus propios datos. | Probabilidad >= umbral. | Probabilidad <= 1 - umbral. |

### Cómo funcionan las estrategias

A continuación se describen de forma sencilla:

#### Momentum
Se calcula el **Relative Strength Index (RSI)**. Cuando el RSI es alto y el
flujo de órdenes (OFI) muestra compras, se envía una orden de compra. Si el RSI
es bajo y el OFI muestra ventas, se vende. Es una estrategia de seguimiento de
tendencia.

#### Mean Reversion
Supone que los precios tienden a volver a una media. Se compra cuando el RSI
cae por debajo de un límite inferior (sobreventa) y se vende cuando supera un
límite superior (sobrecompra).

#### Breakout ATR
Construye **canales de Keltner** usando una media exponencial y el
**Average True Range** (ATR). Si el precio rompe la banda superior se compra;
si perfora la inferior se vende.

#### Breakout Vol
Calcula la media y la desviación estándar de los últimos cierres. Si el precio
se aleja más de `mult` veces la desviación se considera una ruptura.

#### Order Flow
Mide el **Order Flow Imbalance**: cuánta cantidad se ejecuta al bid versus al
ask. Si hay más compras que ventas durante varias velas se compra, y viceversa.

#### Mean Rev OFI
Toma el desequilibrio de flujo de órdenes, lo estandariza (z-score) y aplica un
filtro de volatilidad. Si la presión compradora es muy alta pero la volatilidad
es baja se interpreta como exceso y se vende, esperando que vuelva al promedio.

#### Depth Imbalance
Analiza cuántas órdenes hay en el lado de compra y de venta del libro. Si los
bids pesan mucho más que los asks, se asume posible subida y se compra.

#### Liquidity Events
Detecta eventos donde desaparece la liquidez de un lado del libro (vacío) o
aparece un gran espacio entre el primer y segundo nivel (gap). Reacciona en la
dirección del evento.

#### Cash & Carry
Compara el precio spot con el del futuro perpetuo y considera la tasa de
funding. Si el futuro está por encima del spot y el funding es positivo, conviene
comprar spot y vender perp; la operación inversa se realiza cuando el futuro
está por debajo y el funding es negativo.

#### Cross Exchange Arbitrage
Similar al Cash & Carry pero cruzando dos exchanges distintos. Cuando el precio
del perp excede al spot por encima del umbral se ejecutan órdenes opuestas para
capturar la diferencia.

#### Triangular Arbitrage
Utiliza tres pares (por ejemplo BTC/USDT, ETH/USDT y ETH/BTC). Evalúa si
realizar el ciclo A→B→C→A deja más de lo que se invirtió. Si hay ganancia neta
se genera señal en la dirección favorable.

#### Arbitrage
Es un esqueleto sin lógica concreta. Sirve como punto de partida para que el
usuario implemente su propia estrategia de arbitraje.

#### Triple Barrier
Genera etiquetas según si el precio alcanza primero una barrera superior o
inferior dentro de un horizonte de tiempo. Con estas etiquetas entrena un
modelo de **Gradient Boosting** que luego predice si conviene comprar o vender.

#### ML Strategy
Permite cargar un modelo de *machine learning* entrenado previamente. En cada
barra se calculan características y el modelo devuelve la probabilidad de
subida. Se compra cuando esa probabilidad supera un umbral.

## 6. Panel web y API

Con los contenedores levantados (`./entrypoint.sh`) puedes acceder a:

* **API y panel**: <http://localhost:8000>
* **Prometheus**: <http://localhost:9090>
* **Grafana**: <http://localhost:3000> (usuario y contraseña `admin`)

Desde el panel web es posible ejecutar los mismos comandos CLI y ver métricas
en la pestaña de monitoreo.

## 7. Ejecución de pruebas

El proyecto incluye tests automáticos. Para correrlos utiliza:

```bash
pytest
```

En entornos con pocos recursos puedes ejecutar una versión rápida:

```bash
PYTHONPATH=src:. pytest tests/test_smoke.py
```

## 8. Buenas prácticas y próximos pasos

* Comienza siempre con **backtests** y **paper trading** antes de arriesgar
  dinero real.
* Revisa los logs en la consola o en el archivo correspondiente para entender
  qué hace el bot.
* Ajusta los parámetros de las estrategias según tu tolerancia al riesgo.
* Considera habilitar la persistencia en TimescaleDB para guardar señales y
  resultados a largo plazo.

Con esta guía ya cuentas con todos los elementos para instalar, configurar y
utilizar TradeBot desde cero. ¡Éxitos en tus pruebas y que operes con
responsabilidad!
