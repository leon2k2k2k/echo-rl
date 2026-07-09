#!/usr/bin/env bash
set -euo pipefail

# Submit plain-RL and ECHO pass@16 baselines on the Endless-only seed dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export ENDLESS_SEED_ROOT="${ENDLESS_SEED_ROOT:-$ECHO_RUNTIME_ROOT/data/endless_seed_echo}"
export TERMINAL_AGENT_TRAIN_PARQUET="${TERMINAL_AGENT_TRAIN_PARQUET:-$ENDLESS_SEED_ROOT/train.parquet}"
export TERMINAL_AGENT_VAL_PARQUET="${TERMINAL_AGENT_VAL_PARQUET:-$ENDLESS_SEED_ROOT/val.parquet}"
export TERMINAL_AGENT_DATASET_LABEL="${TERMINAL_AGENT_DATASET_LABEL:-endless_seed_echo}"

exec bash scripts/submit_pass16_baselines.sh
