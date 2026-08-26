#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

python3 -m openmmla.commands.collect.video --project-dir "${PROJECT_DIR}" "$@"
