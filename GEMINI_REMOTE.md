# Proyecto TradeBot: Contexto para Gemini

Este documento sirve como una fuente central de conocimiento sobre el proyecto TradeBot. Su propósito es proporcionar un contexto completo y detallado para facilitar futuras interacciones, desarrollos y modificaciones, asegurando que se respeten los principios de diseño y los requisitos operativos del sistema.

## 1. Resumen General

El proyecto `TradeBot` es un framework avanzado y profesional para desarrollar, probar y ejecutar estrategias de trading algorítmico. No es un simple bot, sino una plataforma completa que soporta:

*   **Múltiples Exchanges**: Capacidad para conectarse a diversos exchanges de criptomonedas (Binance, Bybit, OKX, Deribit, etc.).
*   **Múltiples Modos de Operación**: Soporta trading en vivo (Live), simulación con datos en tiempo real (Paper Trading) y pruebas en entornos de Testnet.
*   **Backtesting Riguroso**: Permite probar estrategias contra datos históricos almacenados en bases de datos de series temporales de alto rendimiento (QuestDB, TimescaleDB).
*   **Gestión de Riesgo Avanzada**: Incorpora módulos específicos para controlar el riesgo de las operaciones antes y durante su ejecución.
*   **Monitorización y Alertas**: Incluye un stack de monitorización profesional (Prometheus, Grafana, Alertmanager) para observar el rendimiento y la salud del sistema en tiempo real.
*   **Infraestructura como Código**: Utiliza Docker y Docker Compose para una gestión, despliegue y escalabilidad consistentes del entorno.

## 2. Componentes Principales y su Interacción

El proyecto está dividido en varios dominios lógicos, cada uno con una responsabilidad clara:

*   **`src/tradingbot` (El Núcleo)**: Contiene toda la lógica de negocio: motores de trading (backtesting, live, paper), implementaciones de estrategias, conectores a exchanges (Adapters), gestión de órdenes y estado del portafolio. Es el orquestador central.
*   **`tests/` (Suite de Pruebas)**: Una colección exhaustiva de pruebas unitarias y de integración que garantizan la fiabilidad y correctitud de cada componente individual y del sistema en su conjunto.
*   **`bin/` (Scripts y Herramientas)**: Scripts auxiliares para tareas como la descarga de datos históricos (`download_history.py`) o el arranque de la infraestructura (`start_stack.sh`).
*   **`data/` (Configuración)**: Archivos de configuración (ej. `backtest.yaml`) que definen el comportamiento del bot (estrategia, parámetros, mercado) sin necesidad de modificar el código.
*   **`db/` y `sql/` (Base de Datos)**: Esquemas y configuraciones para las bases de datos de series temporales, optimizadas para almacenar y consultar grandes volúmenes de datos de mercado.
*   **`monitoring/` (Observabilidad)**: Sistema completo de monitorización para recolectar métricas (Prometheus), visualizarlas (Grafana) y generar alertas (Alertmanager).
*   **`Dockerfile`, `docker-compose.yml`, `Makefile` (DevOps)**: Definen la infraestructura como código, permitiendo empaquetar, orquestar y ejecutar todo el ecosistema de servicios de forma reproducible con comandos sencillos.

## 3. Flujo de Funcionamiento

Un ciclo de vida típico de una señal de trading en el sistema es el siguiente:

1.  **Inicio**: Se levantan todos los servicios (bot, DB, monitorización) a través de `docker-compose up`.
2.  **Ingesta de Datos**: El bot se conecta al exchange. En modo **Live/Paper**, recibe datos en tiempo real vía WebSocket. En modo **Backtesting**, lee datos históricos desde la base de datos local.
3.  **Análisis de Estrategia**: Los datos de mercado fluyen hacia la estrategia configurada. La estrategia analiza los datos y, si sus condiciones se cumplen, genera una **señal** con una **fuerza** asociada.
4.  **Validación de Riesgo**: La señal pasa por el **Gestor de Riesgo**, que verifica si la operación cumple con las reglas predefinidas.
5.  **Dimensionamiento de Posición**: Si el riesgo es aceptable, el módulo de portafolio determina el tamaño de la posición basándose en la **fuerza de la señal**.
6.  **Ejecución**: El **Motor de Ejecución** traduce la señal en una **orden límite** y la envía al exchange (o al libro de órdenes simulado en Paper Trading).
7.  **Monitorización y Gestión**: El sistema monitoriza el estado de la orden (parcialmente completada, completada, etc.) y la posición resultante, registrando métricas y logs continuamente.

## 4. Análisis de Estrategias Implementadas

El bot cuenta con un arsenal de 16 lógicas de trading y metodologías de validación.

### Grupo 1: Estrategias Clásicas (OHLCV)
1.  **Seguimiento de Tendencia**: Usa cruces de medias móviles para seguir tendencias largas.
2.  **Reversión a la Media**: Opera contra movimientos extremos esperando un retorno a la media (ej. con Bandas de Bollinger).
3.  **Momento**: Entra en la dirección de movimientos de precios fuertes y rápidos (ej. con RSI, ROC).
4.  **Scalp Ping-Pong**: Scalping de rango puro entre dos niveles de precios fijos.
5.  **Breakout con Volumen**: Entra en rupturas de rangos confirmadas por un aumento significativo del volumen.
6.  **Breakout con ATR**: Usa el ATR para establecer stops y profits dinámicos tras una ruptura.

### Grupo 2: Estrategias de Arbitraje
7.  **Arbitraje Inter-Exchange**: Compra barato en un exchange y vende caro en otro simultáneamente.
8.  **Arbitraje Triangular**: Explota ineficiencias de tipos de cambio entre 3 pares en un mismo exchange.
9.  **Cash and Carry (Arbitraje de Base)**: Compra en spot y vende en futuros para capturar la diferencia de precio ("la base").

### Grupo 3: Estrategias de Microestructura de Mercado (HFT)
10. **Desbalance del Libro de Órdenes**: Predice movimientos a corto plazo analizando la presión de compra/venta en el libro de órdenes.
11. **Flujo de Órdenes**: Analiza el flujo de trades ejecutados para medir la agresividad de compradores vs. vendedores.

### Grupo 4: Estrategias Cuantitativas y Meta-Estrategias
12. **Estrategia de Machine Learning**: Usa un modelo de ML entrenado para predecir la dirección del precio.
13. **Señales Compuestas**: Combina las señales de múltiples estrategias mediante un sistema de votación o ponderación para mayor robustez.
14. **Método de la Triple Barrera**: Un sistema avanzado de gestión de salidas con take profit, stop loss y un límite de tiempo para cada operación.

### Grupo 5: Metodologías de Validación Avanzada
15. **Purged K-Fold Cross-Validation**: Método robusto de backtesting para prevenir sobreajuste, especialmente en modelos de ML.
16. **Walk-Forward Optimization**: Proceso de backtesting y optimización que simula de forma más realista cómo se re-calibraría una estrategia en un entorno real.

## 5. Mandatos Operativos Clave (Reglas Fundamentales)

Estos son los 7 principios fundamentales que deben regir toda la operativa del bot y cualquier modificación o desarrollo futuro.

**1. Fuerza de la Señal y Tamaño de Posición Dinámico**
*   **Mandato:** Las señales deben emitir un índice de fuerza (ej. de 0.0 a 1.0). El módulo de dimensionamiento de posición usará esta fuerza para determinar el capital a asignar, pudiendo llegar al 100% del capital disponible si la señal es de máxima confianza.
*   **Implicación para Gemini:** Al crear o modificar estrategias, la salida no debe ser un simple `BUY`/`SELL`, sino una tupla `(SEÑAL, FUERZA)`. El código de gestión de portafolio debe implementar la lógica `capital_a_usar = capital_disponible * fuerza_de_la_señal`.

**2. Estrategias Adaptativas y Autocalibrables**
*   **Mandato:** Las estrategias deben ser "inteligentes", adaptándose automáticamente al timeframe en el que operan y autocalibrando sus parámetros según las condiciones del mercado (ej. volatilidad, tendencia).
*   **Implicación para Gemini:** Evitar parámetros fijos (hardcoded). Las estrategias deben incluir lógica para, por ejemplo, usar períodos de indicadores más largos en timeframes mayores o ajustar umbrales basados en el ATR actual. La meta es la robustez a través de diferentes regímenes de mercado.

**3. Uso de Órdenes Límite y Gestión Avanzada**
*   **Mandato:** Se deben usar órdenes límite (`LIMIT`) por defecto para evitar slippage. El sistema debe ser capaz de gestionar órdenes parcialmente completadas (`partial fills`) y tener una lógica para decidir si una orden no completada debe ser cancelada y reposicionada si la señal original sigue siendo válida.
*   **Implicación para Gemini:** El motor de ejecución debe estar diseñado en torno a órdenes límite. Se necesita un "gestor de órdenes" que monitorice el estado de las órdenes abiertas y las compare con las señales actuales de la estrategia para tomar decisiones de cancelación/reposicionamiento.

**4. Escalado y Desescalado de Posiciones (Pyramiding & Scaling Out)**
*   **Mandato:** El bot debe poder añadir capital a posiciones ganadoras (pyramiding) si nuevas señales confirman la dirección. Inversamente, debe poder reducir la posición (scaling out) si la tendencia o las señales pierden fuerza, antes de un cierre completo.
*   **Implicación para Gemini:** La gestión de posición no es binaria (abierta/cerrada). El sistema debe permitir múltiples órdenes sobre una misma posición para incrementar o decrementar su tamaño. La fuerza de la señal (Mandato 1) es clave aquí para decidir cuánto añadir o quitar.

**5. Restricción de Venta en Corto (Short) para Mercados Spot**
*   **Mandato:** Si el bot opera en un mercado `spot`, la lógica debe prohibir explícitamente la apertura de posiciones en corto (`SHORT`).
*   **Implicación para Gemini:** Antes de enviar una orden, el motor de ejecución o el gestor de riesgo debe verificar el tipo de mercado (`venue_kind`). Si es `spot` y la señal es `SELL` para abrir posición, debe ser bloqueada.

**6. `risk_pct` como Stop Loss de Emergencia**
*   **Mandato:** El parámetro `risk_pct` debe funcionar exclusivamente como un stop loss de protección contra movimientos adversos y repentinos. No debe limitar el capital inicial con el que se abre una operación (eso lo define la fuerza de la señal, Mandato 1).
*   **Implicación para Gemini:** La lógica de dimensionamiento de posición debe ignorar `risk_pct`. Este parámetro solo debe usarse para crear una orden `STOP` una vez que la posición ya está abierta.

**7. Operativa de Alta Frecuencia**
*   **Mandato:** El diseño general debe favorecer la operativa de alta frecuencia. El bot debe estar buscando y ejecutando operaciones constantemente.
*   **Implicación para Gemini:** Priorizar la eficiencia y la baja latencia en todo el código. Utilizar conexiones WebSocket, procesamiento asíncrono y evitar cálculos pesados en el bucle principal. Las estrategias de microestructura de mercado y arbitraje son de especial interés.

