#!/usr/bin/env bash
set -e
SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."
docker compose up -d

# Wait for TimescaleDB to be ready
until pg_isready -h localhost -p 5432 -U postgres >/dev/null 2>&1; do
  sleep 1
done

# Apply database schema
PGPASSWORD=postgres psql -h localhost -U postgres -d trading -f sql/schema_timescale.sql

# Wait for QuestDB to be ready
until curl -f http://localhost:9000/health >/dev/null 2>&1; do
  sleep 1
done

# Apply QuestDB schema
psql -h localhost -p 8812 -U admin -d qdb -f sql/schema_quest.sql
