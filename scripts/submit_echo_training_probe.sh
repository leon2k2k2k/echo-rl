#!/usr/bin/env bash
set -euo pipefail

# Submit a fast ECHO training-path probe:
# - force synthetic smoke-ok terminal-agent data
# - keep the 4-GPU ECHO/FSDP/vLLM shape
# - use tiny rollout limits so the run reaches policy_train quickly

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export ECHO_SMOKE_DATA_DIR="${ECHO_SMOKE_DATA_DIR:-$ECHO_RUNTIME_ROOT/data}"
export ECHO_DEFAULT_TRAIN_PARQUET="$ECHO_SMOKE_DATA_DIR/terminal_agent_smoke_train.parquet"
export ECHO_DEFAULT_VAL_PARQUET="$ECHO_SMOKE_DATA_DIR/terminal_agent_smoke_val.parquet"

# Force the synthetic smoke dataset even when alex_cluster.env points at real
# terminal-agent/TMax parquets.
export TERMINAL_AGENT_TRAIN_PARQUET="$ECHO_DEFAULT_TRAIN_PARQUET"
export TERMINAL_AGENT_VAL_PARQUET="$ECHO_DEFAULT_VAL_PARQUET"

ensure_terminal_agent_smoke_data
check_terminal_agent_data_files

export RUN_KIND="${RUN_KIND:-echo-4gpu-training-probe}"
export RUN_ID="${RUN_ID:-${RUN_KIND}-$(date +%Y%m%d-%H%M%S)}"
export CONFIG_PATH="${CONFIG_PATH:-echo_configs/qwen3_8b_rl_echo_4gpu_micro_smoke.yaml}"
export ECHO_HOLD_ON_FAILURE="${ECHO_HOLD_ON_FAILURE:-1}"
export ECHO_HOLD_ON_FAILURE_SECONDS="${ECHO_HOLD_ON_FAILURE_SECONDS:-3600}"

partition="${PARTITION:-a01}"

echo "Submitting ECHO training probe"
echo "PARTITION=$partition"
echo "RUN_KIND=$RUN_KIND"
echo "RUN_ID=$RUN_ID"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "TERMINAL_AGENT_TRAIN_PARQUET=$TERMINAL_AGENT_TRAIN_PARQUET"
echo "TERMINAL_AGENT_VAL_PARQUET=$TERMINAL_AGENT_VAL_PARQUET"
echo "ECHO_HOLD_ON_FAILURE=$ECHO_HOLD_ON_FAILURE"
echo "ECHO_HOLD_ON_FAILURE_SECONDS=$ECHO_HOLD_ON_FAILURE_SECONDS"

job_id="$(sbatch -p "$partition" --parsable slurm/echo_nextlat_smoke.sbatch)"
echo "J=$job_id"
echo "Logs:"
echo "  logs/echo-rl-smoke-$job_id.out"
echo "  logs/echo-rl-smoke-$job_id.err"
