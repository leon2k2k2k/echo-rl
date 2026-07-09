#!/usr/bin/env bash
set -euo pipefail

# Submit a smoke run on the ubuntu:22.04-only public seed dataset.
#
# Intentionally pins TERMINAL_AGENT_* dataset variables. Do not inherit those
# values from the caller: stale smoke parquet paths are too easy to leak into
# preflights.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

UBUNTU22_SEED_ROOT="$ECHO_RUNTIME_ROOT/data/ubuntu22_seed_echo"
export UBUNTU22_SEED_ROOT
export PUBLIC_SEED_ROOT="$UBUNTU22_SEED_ROOT"
export TERMINAL_AGENT_TRAIN_PARQUET="$UBUNTU22_SEED_ROOT/train.parquet"
export TERMINAL_AGENT_VAL_PARQUET="$UBUNTU22_SEED_ROOT/val.parquet"
export TERMINAL_AGENT_DATASET_LABEL="ubuntu22_seed_echo"

exec bash scripts/submit_public_seed_smoke.sh
