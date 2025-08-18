#!/usr/bin/env bash
set -e
SCRIPT_DIR=$(cd -- "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR/.."
docker compose -f sql/docker-compose.timescale.yml up -d
