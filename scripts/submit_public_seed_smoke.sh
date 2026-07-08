#!/usr/bin/env bash
set -euo pipefail

# Submit one tiny terminal-agent smoke on the public combined seed dataset.
# This is meant as a cheap preflight before longer public-seed training runs.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export PUBLIC_SEED_ROOT="${PUBLIC_SEED_ROOT:-$ECHO_RUNTIME_ROOT/data/public_seed_echo}"
export TERMINAL_AGENT_TRAIN_PARQUET="${TERMINAL_AGENT_TRAIN_PARQUET:-$PUBLIC_SEED_ROOT/train.parquet}"
export TERMINAL_AGENT_VAL_PARQUET="${TERMINAL_AGENT_VAL_PARQUET:-$PUBLIC_SEED_ROOT/val.parquet}"
export TERMINAL_AGENT_DATASET_LABEL="${TERMINAL_AGENT_DATASET_LABEL:-public_seed_echo_combined}"
export PARTITION="${PARTITION:-a01}"
export SBATCH_GRES="${SBATCH_GRES:-gpu:4}"
export SBATCH_TIME="${SBATCH_TIME:-02:00:00}"
export SMOKE_VARIANT="${SMOKE_VARIANT:-plain}"
export ECHO_LOG_DIR="${ECHO_LOG_DIR:-$ECHO_RUNTIME_ROOT/logs}"
export ECHO_HOLD_ON_FAILURE="${ECHO_HOLD_ON_FAILURE:-1}"
export ECHO_HOLD_ON_FAILURE_SECONDS="${ECHO_HOLD_ON_FAILURE_SECONDS:-3600}"
export ECHO_LOG_POLICY_TRAIN_METRICS="${ECHO_LOG_POLICY_TRAIN_METRICS:-1}"
export SKIP_ECHO_RUNTIME_IMPORT_CHECK="${SKIP_ECHO_RUNTIME_IMPORT_CHECK:-1}"
mkdir -p "$ECHO_LOG_DIR"

check_terminal_agent_data_files

case "$SMOKE_VARIANT" in
  plain)
    export RUN_KIND="${TERMINAL_AGENT_DATASET_LABEL}-plain-smoke"
    export CONFIG_PATH="echo_configs/qwen3_8b_rl_plain_smoke.yaml"
    ;;
  echo)
    export RUN_KIND="${TERMINAL_AGENT_DATASET_LABEL}-echo-smoke"
    export CONFIG_PATH="echo_configs/qwen3_8b_rl_echo_smoke.yaml"
    ;;
  nextlat)
    export RUN_KIND="${TERMINAL_AGENT_DATASET_LABEL}-nextlat-smoke"
    export CONFIG_PATH="echo_configs/qwen3_8b_rl_nextlat_smoke.yaml"
    ;;
  *)
    echo "error: SMOKE_VARIANT must be plain, echo, or nextlat; got $SMOKE_VARIANT" >&2
    exit 2
    ;;
esac

if [[ "${SKIP_ECHO_RUNTIME_IMPORT_CHECK:-0}" != "1" ]]; then
  "$PYTHON" scripts/check_echo_runtime_imports.py
else
  echo "Skipping ECHO runtime import check because SKIP_ECHO_RUNTIME_IMPORT_CHECK=1"
fi

echo "Public seed smoke preflight"
echo "TERMINAL_AGENT_DATASET_LABEL=$TERMINAL_AGENT_DATASET_LABEL"
echo "PUBLIC_SEED_ROOT=$PUBLIC_SEED_ROOT"
echo "TERMINAL_AGENT_TRAIN_PARQUET=$TERMINAL_AGENT_TRAIN_PARQUET"
echo "TERMINAL_AGENT_VAL_PARQUET=$TERMINAL_AGENT_VAL_PARQUET"
echo "SMOKE_VARIANT=$SMOKE_VARIANT"
echo "RUN_KIND=$RUN_KIND"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "SKIP_ECHO_RUNTIME_IMPORT_CHECK=$SKIP_ECHO_RUNTIME_IMPORT_CHECK"
echo "PARTITION=$PARTITION"
if [[ -n "${SBATCH_NODE:-}" ]]; then
  echo "SBATCH_NODE=$SBATCH_NODE"
fi
echo "SBATCH_GRES=$SBATCH_GRES"
echo "SBATCH_TIME=$SBATCH_TIME"

if [[ "${PRINT_DATASET_SAMPLE:-1}" == "1" ]]; then
  "$PYTHON" scripts/inspect_echo_parquet.py "$TERMINAL_AGENT_VAL_PARQUET" --rows 1 || true
fi

stamp="$(date +%Y%m%d-%H%M%S)"
export RUN_ID="${RUN_KIND}-${stamp}"

sbatch_args=(
  -p "$PARTITION"
  --gres="$SBATCH_GRES"
  --time="$SBATCH_TIME"
  --output="$ECHO_LOG_DIR/%x-%j.out"
  --error="$ECHO_LOG_DIR/%x-%j.err"
  --parsable
)
if [[ -n "${SBATCH_NODE:-}" ]]; then
  sbatch_args+=(--nodelist="$SBATCH_NODE")
fi
if [[ "${SBATCH_EXCLUSIVE:-0}" == "1" ]]; then
  sbatch_args+=(--exclusive)
fi

job_id="$(
  sbatch "${sbatch_args[@]}" slurm/echo_nextlat_smoke.sbatch
)"

echo
echo "Submitted $RUN_KIND job $job_id"
echo "Log files:"
echo "  $ECHO_LOG_DIR/echo-rl-smoke-$job_id.out"
echo "  $ECHO_LOG_DIR/echo-rl-smoke-$job_id.err"

cat <<EOF

Monitor:
  cd $ECHO_REPO
  squeue -j $job_id -o "%.18i %.10P %.30j %.8T %.10M %.20N %.30R"
  tail -f "$ECHO_LOG_DIR/echo-rl-smoke-$job_id.out" "$ECHO_LOG_DIR/echo-rl-smoke-$job_id.err"

Metrics:
  cd $ECHO_REPO
  bash scripts/grep_smoke_metrics.sh $job_id
EOF
