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
if [[ "${SKIP_ECHO_RUNTIME_IMPORT_CHECK:-0}" != "1" ]]; then
  "$PYTHON" scripts/check_echo_runtime_imports.py
else
  echo "Skipping ECHO runtime import check because SKIP_ECHO_RUNTIME_IMPORT_CHECK=1"
fi

# Harbor task Dockerfiles rely on locally cached base images on some Alex nodes.
# BuildKit may still perform remote metadata lookups even when the image tag
# exists locally, so use the classic builder for smoke task environments.
export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-0}"
export ECHO_CACHE_TASK_IMAGES="${ECHO_CACHE_TASK_IMAGES:-1}"

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
echo "DOCKER_BUILDKIT=$DOCKER_BUILDKIT"
echo "COMPOSE_DOCKER_CLI_BUILD=$COMPOSE_DOCKER_CLI_BUILD"
echo "ECHO_CACHE_TASK_IMAGES=$ECHO_CACHE_TASK_IMAGES"

"$PYTHON" -m echo_rl.terminal_agent.entrypoint --config "$CONFIG_PATH"
