#!/usr/bin/env bash
set -euo pipefail

# Build an ECHO-compatible dataset from Endless Terminals only.
# This avoids OpenThoughts tasks that currently require ubuntu:24.04.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export DATA_ROOT="${DATA_ROOT:-$ECHO_RUNTIME_ROOT/data/public_seed_sources}"
export ENDLESS_DIR="${ENDLESS_DIR:-$DATA_ROOT/endless-terminals}"
export OUT_ROOT="${OUT_ROOT:-$ECHO_RUNTIME_ROOT/data/endless_seed_echo}"
export VAL_SIZE="${VAL_SIZE:-100}"
export SEED="${SEED:-0}"
export MAX_ENDLESS="${MAX_ENDLESS:-0}"

if [[ ! -d "$ENDLESS_DIR" ]]; then
  cat >&2 <<EOF
error: Endless Terminals snapshot not found: $ENDLESS_DIR

Download it first, for example:
  hf download obiwan96/endless-terminals --repo-type dataset --local-dir "$ENDLESS_DIR"
EOF
  exit 1
fi

mkdir -p "$OUT_ROOT"

build_args=(
  --endless-dir "$ENDLESS_DIR"
  --train-out "$OUT_ROOT/train.parquet"
  --val-out "$OUT_ROOT/val.parquet"
  --val-size "$VAL_SIZE"
  --seed "$SEED"
)
if [[ "$MAX_ENDLESS" != "0" ]]; then
  build_args+=(--max-endless "$MAX_ENDLESS")
fi

"$PYTHON" scripts/build_public_seed_echo_dataset.py "${build_args[@]}"

echo
echo "Inspecting validation sample:"
"$PYTHON" scripts/inspect_echo_parquet.py "$OUT_ROOT/val.parquet" --rows 1 || true

echo
echo "Docker requirement summary:"
"$PYTHON" scripts/summarize_task_docker_requirements.py "$OUT_ROOT/val.parquet" --rows "$VAL_SIZE" --examples 3 || true

cat <<EOF

Endless-only dataset ready:
  train: $OUT_ROOT/train.parquet
  val:   $OUT_ROOT/val.parquet
EOF
