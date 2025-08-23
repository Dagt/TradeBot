#!/usr/bin/env bash
set -e
SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."
docker compose -f sql/docker-compose.timescale.yml up -d

# Wait for TimescaleDB to be ready
until pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; do
  sleep 1
done

# Apply database schema
PGPASSWORD=postgres psql -h localhost -U postgres -d trading -f sql/schema_timescale.sql
