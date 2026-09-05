#!/usr/bin/env bash
# Fetch every model the pipeline needs (~90 GB).
set -euo pipefail
cd "$(dirname "$0")/.."
uv run odistil check --download
