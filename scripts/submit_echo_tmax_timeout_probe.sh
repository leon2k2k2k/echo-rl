#!/usr/bin/env bash
set -euo pipefail

# Submit a fast real-TMax ECHO probe:
# - uses cached TMax smoke parquets/images
# - keeps the 4-GPU ECHO/FSDP/vLLM shape
# - sets terminal-agent timeout to 60s via qwen3_8b_rl_echo_fast_timeout_smoke.yaml
# - keeps failed allocations alive for postmortem artifacts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

# Default to the real TMax smoke data for this probe, but allow intentional
# overrides without accidentally inheriting stale TERMINAL_AGENT_* paths from
# the user's shell.
export TERMINAL_AGENT_TRAIN_PARQUET="${ECHO_PROBE_TRAIN_PARQUET:-$ECHO_RUNTIME_ROOT/data/tmax_echo/train.parquet}"
export TERMINAL_AGENT_VAL_PARQUET="${ECHO_PROBE_VAL_PARQUET:-$ECHO_RUNTIME_ROOT/data/tmax_echo/val.parquet}"
check_terminal_agent_data_files

export RUN_KIND="${RUN_KIND:-echo-tmax-timeout-probe}"
export RUN_ID="${RUN_ID:-${RUN_KIND}-$(date +%Y%m%d-%H%M%S)}"
export CONFIG_PATH="${CONFIG_PATH:-echo_configs/qwen3_8b_rl_echo_fast_timeout_smoke.yaml}"
export ECHO_HOLD_ON_FAILURE="${ECHO_HOLD_ON_FAILURE:-1}"
export ECHO_HOLD_ON_FAILURE_SECONDS="${ECHO_HOLD_ON_FAILURE_SECONDS:-3600}"

partition="${PARTITION:-a01}"

echo "Submitting fast TMax timeout ECHO probe"
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
