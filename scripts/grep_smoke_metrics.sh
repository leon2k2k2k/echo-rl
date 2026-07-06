#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == "" ]]; then
  echo "usage: $0 JOBID" >&2
  exit 2
fi

job_id="$1"
shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/cluster_env.sh"

log_glob=(
  "$ECHO_RUNTIME_ROOT"/logs/*-"$job_id".out
  "$ECHO_RUNTIME_ROOT"/logs/*-"$job_id".err
  "$REPO_ROOT"/logs/*-"$job_id".out
  "$REPO_ROOT"/logs/*-"$job_id".err
)

grep -Eh \
  "RUN_KIND=|CONFIG_PATH=|OUTPUT_DIR=|Step [0-9]+:|policy_loss|final_loss|grad_norm|world_loss|world_ce|world_tokens|nextlat_|Traceback|RuntimeError|OutOfMemory|CUDA out of memory|FAILED|ModuleNotFound|FileNotFound" \
  "${log_glob[@]}" "$@" | tail -200
