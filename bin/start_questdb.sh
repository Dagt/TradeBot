#!/usr/bin/env bash
set -e
SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."
docker compose -f sql/docker-compose.questdb.yml up -d

# Wait for QuestDB to be ready
until curl -f http://localhost:9000/health >/dev/null 2>&1; do
  sleep 1
done

# Apply database schema
psql -h localhost -p 8812 -U admin -d qdb -f sql/schema_quest.sql
