# Gestión de claves API

Mantener seguras las credenciales es fundamental. Algunas recomendaciones:

- Usa un archivo `.env` (no versionado) o `~/.secrets` para almacenar claves.
- Limita los permisos de las API keys al mínimo necesario y deshabilita retiros.
- Emplea la clase `SecretsManager` para cargar y rotar múltiples claves, respetando *rate limits*.
- El gestor sólo registra el identificador de la clave (últimos 4 caracteres) para evitar filtrar información sensible.
- Revisa periódicamente los registros y rota las claves con regularidad.
- Nunca compartas claves ni las publiques en repositorios.

El comando `real-run` exige el flag `--i-know-what-im-doing` como medida de
protección para evitar operar por accidente con fondos reales.

Algunas funciones avanzadas (API web, panel de monitoreo, estrategias de
arbitraje) también requieren considerar permisos y autenticación. Consulta
[extra_features.md](extra_features.md) para más detalles.
