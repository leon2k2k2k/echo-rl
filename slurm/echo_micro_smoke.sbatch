#!/usr/bin/env bash
#SBATCH --job-name=echo-rl-micro
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:45:00

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

# shellcheck source=/dev/null
source scripts/cluster_env.sh

run_kind="${RUN_KIND:-echo-micro}"
run_id="${RUN_ID:-${run_kind}-smoke-${SLURM_JOB_ID:-local}}"
export OUTPUT_DIR="${OUTPUT_DIR:-$ECHO_RUNTIME_ROOT/outputs/$run_id}"
export CONFIG_PATH="${CONFIG_PATH:-echo_configs/qwen3_8b_rl_echo_micro_smoke.yaml}"
export ECHO_POLICY_RANK_LOG_DIR="${ECHO_POLICY_RANK_LOG_DIR:-$OUTPUT_DIR/policy_rank_logs}"
mkdir -p "$OUTPUT_DIR" "$OUTPUT_DIR/debug"

# shellcheck source=/dev/null
source scripts/slurm_debug_logging.sh
trap 'rc=$?; slurm_debug_exit_trap "$OUTPUT_DIR" "$rc"; exit "$rc"' EXIT

print_slurm_preamble
start_slurm_heartbeat "$OUTPUT_DIR"

ensure_terminal_agent_smoke_data
check_terminal_agent_data_files
if [[ "${SKIP_ECHO_RUNTIME_IMPORT_CHECK:-0}" != "1" ]]; then
  "$PYTHON" scripts/check_echo_runtime_imports.py
else
  echo "Skipping ECHO runtime import check because SKIP_ECHO_RUNTIME_IMPORT_CHECK=1"
fi

export DOCKER_BUILDKIT="${DOCKER_BUILDKIT:-0}"
export COMPOSE_DOCKER_CLI_BUILD="${COMPOSE_DOCKER_CLI_BUILD:-0}"
export ECHO_CACHE_TASK_IMAGES="${ECHO_CACHE_TASK_IMAGES:-1}"
export ECHO_DOCKER_WHEELHOUSE="${ECHO_DOCKER_WHEELHOUSE:-$ECHO_RUNTIME_ROOT/docker_wheelhouse}"
export ECHO_DOCKER_PIP_INDEX_URL="${ECHO_DOCKER_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export ECHO_DOCKER_BASE_IMAGE="${ECHO_DOCKER_BASE_IMAGE:-ubuntu:22.04}"
export ECHO_DOCKER_BASE_IMAGE_TAR="${ECHO_DOCKER_BASE_IMAGE_TAR:-$ECHO_RUNTIME_ROOT/docker_cache/ubuntu_22_04.tar}"
export ECHO_DOCKER_LOAD_BASE_IMAGE="${ECHO_DOCKER_LOAD_BASE_IMAGE:-1}"
export ECHO_REQUIRE_DOCKER_BASE_IMAGE="${ECHO_REQUIRE_DOCKER_BASE_IMAGE:-1}"
export ECHO_DOCKER_IMAGE_TAR="${ECHO_DOCKER_IMAGE_TAR:-$ECHO_RUNTIME_ROOT/docker_cache/tmax_smoke_images.tar}"
export ECHO_DOCKER_LOAD_IMAGES="${ECHO_DOCKER_LOAD_IMAGES:-1}"

if [[ "$ECHO_DOCKER_LOAD_BASE_IMAGE" == "1" && -f "$ECHO_DOCKER_BASE_IMAGE_TAR" ]]; then
  echo "Loading Docker base image cache from $ECHO_DOCKER_BASE_IMAGE_TAR"
  docker load -i "$ECHO_DOCKER_BASE_IMAGE_TAR"
elif [[ -n "${ECHO_DOCKER_BASE_IMAGE_TAR:-}" ]]; then
  echo "Docker base image cache not loaded: $ECHO_DOCKER_BASE_IMAGE_TAR not found or ECHO_DOCKER_LOAD_BASE_IMAGE=$ECHO_DOCKER_LOAD_BASE_IMAGE"
fi

if [[ "$ECHO_DOCKER_LOAD_IMAGES" == "1" && -f "$ECHO_DOCKER_IMAGE_TAR" ]]; then
  echo "Loading Docker smoke image cache from $ECHO_DOCKER_IMAGE_TAR"
  docker load -i "$ECHO_DOCKER_IMAGE_TAR"
elif [[ -n "${ECHO_DOCKER_IMAGE_TAR:-}" ]]; then
  echo "Docker smoke image cache not loaded: $ECHO_DOCKER_IMAGE_TAR not found or ECHO_DOCKER_LOAD_IMAGES=$ECHO_DOCKER_LOAD_IMAGES"
fi

if docker image inspect "$ECHO_DOCKER_BASE_IMAGE" >/dev/null 2>&1; then
  echo "Docker base image available: $ECHO_DOCKER_BASE_IMAGE"
elif [[ "$ECHO_REQUIRE_DOCKER_BASE_IMAGE" == "1" ]]; then
  echo "Required Docker base image is missing: $ECHO_DOCKER_BASE_IMAGE" >&2
  echo "Expected cache tar: $ECHO_DOCKER_BASE_IMAGE_TAR" >&2
  exit 1
else
  echo "Docker base image missing locally: $ECHO_DOCKER_BASE_IMAGE; Docker may try to pull it during task builds"
fi

bash scripts/sync_echo_to_skyrl.sh

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
echo "ECHO_DOCKER_WHEELHOUSE=$ECHO_DOCKER_WHEELHOUSE"
echo "ECHO_DOCKER_PIP_INDEX_URL=$ECHO_DOCKER_PIP_INDEX_URL"
echo "ECHO_DOCKER_BASE_IMAGE=$ECHO_DOCKER_BASE_IMAGE"
echo "ECHO_DOCKER_BASE_IMAGE_TAR=$ECHO_DOCKER_BASE_IMAGE_TAR"
echo "ECHO_DOCKER_LOAD_BASE_IMAGE=$ECHO_DOCKER_LOAD_BASE_IMAGE"
echo "ECHO_REQUIRE_DOCKER_BASE_IMAGE=$ECHO_REQUIRE_DOCKER_BASE_IMAGE"
echo "ECHO_DOCKER_IMAGE_TAR=$ECHO_DOCKER_IMAGE_TAR"
echo "ECHO_DOCKER_LOAD_IMAGES=$ECHO_DOCKER_LOAD_IMAGES"

echo "ENTRYPOINT_START=$(date -Is)"
"$PYTHON" -m echo_rl.terminal_agent.entrypoint --config "$CONFIG_PATH"
echo "ENTRYPOINT_DONE=$(date -Is)"
