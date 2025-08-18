# Integration Test Scenarios

Este documento describe los escenarios de pruebas de integración añadidos para
simular el flujo completo del bot de trading con datos grabados y para someter
a estrés el motor de backtesting.

## Flujo completo con datos grabados

Se utiliza el conjunto `data/examples/btcusdt_1m.csv` junto a una estrategia que
intenta comprar en cada barra. El `RiskManager` limita la posición máxima a una
unidad, por lo que solo se ejecuta una orden. El resultado esperado es un PnL de
aproximadamente `2.85` unidades, calculado como la diferencia entre el precio
de cierre final y el precio promedio de la orden ejecutada. Cualquier intento de
exceder el límite de riesgo es rechazado.

## Prueba de estrés de latencia y spreads

El segundo escenario ejecuta el mismo flujo bajo un `StressConfig` que duplica la
latencia y multiplica el spread por `2`. El motor debe seguir generando fills y
la latencia reportada en las órdenes debe reflejar el incremento.

Ambas pruebas están marcadas con `@pytest.mark.integration` y se ejecutan
automáticamente en CI mediante `pytest -m integration`.
