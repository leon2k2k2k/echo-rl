#!/usr/bin/env bash
set -euo pipefail

# Submit plain-RL and ECHO pass@16 baselines on the ubuntu:22.04-only seed dataset.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export UBUNTU22_SEED_ROOT="${UBUNTU22_SEED_ROOT:-$ECHO_RUNTIME_ROOT/data/ubuntu22_seed_echo}"
export TERMINAL_AGENT_TRAIN_PARQUET="${TERMINAL_AGENT_TRAIN_PARQUET:-$UBUNTU22_SEED_ROOT/train.parquet}"
export TERMINAL_AGENT_VAL_PARQUET="${TERMINAL_AGENT_VAL_PARQUET:-$UBUNTU22_SEED_ROOT/val.parquet}"
export TERMINAL_AGENT_DATASET_LABEL="${TERMINAL_AGENT_DATASET_LABEL:-ubuntu22_seed_echo}"

exec bash scripts/submit_pass16_baselines.sh
