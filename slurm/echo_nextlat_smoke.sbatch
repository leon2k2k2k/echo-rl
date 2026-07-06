#!/usr/bin/env bash
#SBATCH --job-name=echo-rl-smoke
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# shellcheck source=/dev/null
source scripts/cluster_env.sh

ensure_terminal_agent_smoke_data
check_terminal_agent_data_files

bash scripts/sync_echo_to_skyrl.sh

run_kind="${RUN_KIND:-echo-nextlat}"
run_id="${RUN_ID:-${run_kind}-smoke-${SLURM_JOB_ID:-local}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ECHO_RUNTIME_ROOT/outputs/$run_id}"
export CONFIG_PATH="${CONFIG_PATH:-echo_configs/qwen3_8b_rl_nextlat_smoke.yaml}"
mkdir -p "$OUTPUT_DIR"

cd "$SKYRL_DIR"
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"

echo "RUN_KIND=$run_kind"
echo "RUN_ID=$run_id"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "PYTHON=$PYTHON"
echo "SKYRL_DIR=$SKYRL_DIR"
echo "ECHO_REPO=$ECHO_REPO"

"$PYTHON" -m echo_rl.terminal_agent.entrypoint --config "$CONFIG_PATH"
