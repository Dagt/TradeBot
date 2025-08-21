# Blueprint de Arquitectura

Este documento describe los componentes principales del bot y cómo se
comunican. La intención es ofrecer una visión general accesible para
personas sin experiencia técnica.

## Componentes

### Adaptadores
Conectores que permiten comunicarse con los intercambios (Binance, Bybit,
OKX, etc.). Se encuentran en `src/tradingbot/adapters` y se encargan de
enviar órdenes y recibir información del mercado.

### Ingesta de datos
Los módulos en `src/tradingbot/data` descargan y almacenan información del
mercado. El comando [`ingest`](docs/commands.md#ingest) permite recibir
flujos en vivo y opcionalmente guardarlos en la base de datos.

### Estrategias
Las reglas de decisión para comprar o vender viven en
`src/tradingbot/strategies`. Existen estrategias de rompimiento, arbitraje,
mean reversion y otras. Cada estrategia genera señales que luego se
transforman en órdenes.

### Ejecución
El módulo `src/tradingbot/execution` traduce las señales en órdenes reales y
las envía al intercambio correspondiente mediante un `ExecutionRouter`.

### Gestión de riesgo
En `src/tradingbot/risk` hay componentes que limitan pérdidas, controlan el
apalancamiento y aplican otras reglas de seguridad.

### Almacenamiento
Los datos y las métricas se guardan en TimescaleDB o QuestDB. Las funciones
se encuentran en `src/tradingbot/storage` y se utilizan tanto para ingestar
datos como para generar reportes.

### Monitoreo
La carpeta [`monitoring`](monitoring/) contiene la configuración de Grafana
y Prometheus. Los dashboards disponibles se describen en
[docs/dashboards.md](docs/dashboards.md).

## Flujo básico

1. **Ingesta** de datos de mercado.
2. **Estrategias** generan señales.
3. **Ejecución** convierte señales en órdenes.
4. **Gestión de riesgo** supervisa posiciones y límites.
5. **Monitoreo** muestra el desempeño y el estado del sistema.

Este blueprint sirve como mapa para comprender cómo las piezas encajan entre
sí dentro de TradeBot.
