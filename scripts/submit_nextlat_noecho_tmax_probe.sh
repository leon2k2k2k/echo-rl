#!/usr/bin/env bash
set -euo pipefail

# Submit a real-TMax NextLat-only control:
# - uses cached TMax smoke parquets/images
# - disables ECHO world-token CE loss
# - keeps NextLat auxiliary loss enabled on environment observation tokens
# - keeps the same 4-GPU FSDP/vLLM shape as the ECHO probes

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

# Default to real TMax smoke data. Use ECHO_PROBE_* only for intentional
# overrides; do not inherit stale TERMINAL_AGENT_* paths from the shell.
export TERMINAL_AGENT_TRAIN_PARQUET="${ECHO_PROBE_TRAIN_PARQUET:-$ECHO_RUNTIME_ROOT/data/tmax_echo/train.parquet}"
export TERMINAL_AGENT_VAL_PARQUET="${ECHO_PROBE_VAL_PARQUET:-$ECHO_RUNTIME_ROOT/data/tmax_echo/val.parquet}"
check_terminal_agent_data_files

export RUN_KIND="${RUN_KIND:-nextlat-noecho-tmax-probe}"
export RUN_ID="${RUN_ID:-${RUN_KIND}-$(date +%Y%m%d-%H%M%S)}"
export CONFIG_PATH="${CONFIG_PATH:-echo_configs/qwen3_8b_rl_nextlat_noecho_smoke.yaml}"
export ECHO_HOLD_ON_FAILURE="${ECHO_HOLD_ON_FAILURE:-1}"
export ECHO_HOLD_ON_FAILURE_SECONDS="${ECHO_HOLD_ON_FAILURE_SECONDS:-3600}"

partition="${PARTITION:-a01}"

echo "Submitting real-TMax NextLat-only control"
echo "PARTITION=$partition"
echo "RUN_KIND=$RUN_KIND"
echo "RUN_ID=$RUN_ID"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "TERMINAL_AGENT_TRAIN_PARQUET=$TERMINAL_AGENT_TRAIN_PARQUET"
echo "TERMINAL_AGENT_VAL_PARQUET=$TERMINAL_AGENT_VAL_PARQUET"
echo "ECHO_HOLD_ON_FAILURE=$ECHO_HOLD_ON_FAILURE"
echo "ECHO_HOLD_ON_FAILURE_SECONDS=$ECHO_HOLD_ON_FAILURE_SECONDS"
echo "SBATCH_EXTRA=${SBATCH_EXTRA:-}"

read -r -a sbatch_extra_args <<< "${SBATCH_EXTRA:-}"
job_id="$(sbatch -p "$partition" "${sbatch_extra_args[@]}" --parsable slurm/echo_nextlat_smoke.sbatch)"
echo "J=$job_id"
echo "Logs:"
echo "  logs/echo-rl-smoke-$job_id.out"
echo "  logs/echo-rl-smoke-$job_id.err"
