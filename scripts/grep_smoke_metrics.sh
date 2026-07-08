#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/cluster_env.sh"

if [[ ${1:-} == "" ]]; then
  shopt -s nullglob
  log_glob=()

  if command -v squeue >/dev/null 2>&1; then
    mapfile -t active_jobs < <(
      squeue -h -u "${USER:-alex}" -o "%A %j" \
        | awk '$2 ~ /^echo-rl-/ {print $1}'
    )

    for job_id in "${active_jobs[@]}"; do
      for log in \
        "$ECHO_RUNTIME_ROOT/logs/echo-rl-smoke-$job_id.out" \
        "$ECHO_RUNTIME_ROOT/logs/echo-rl-smoke-$job_id.err" \
        "$REPO_ROOT/logs/echo-rl-smoke-$job_id.out" \
        "$REPO_ROOT/logs/echo-rl-smoke-$job_id.err"
      do
        [[ -f "$log" ]] && log_glob+=("$log")
      done
    done
  fi

  if (( ${#log_glob[@]} == 0 )); then
    candidates=(
      "$ECHO_RUNTIME_ROOT"/logs/echo-rl-smoke-*.out
      "$ECHO_RUNTIME_ROOT"/logs/echo-rl-smoke-*.err
      "$REPO_ROOT"/logs/echo-rl-smoke-*.out
      "$REPO_ROOT"/logs/echo-rl-smoke-*.err
    )
    if (( ${#candidates[@]} > 0 )); then
      mapfile -t log_glob < <(ls -t "${candidates[@]}" 2>/dev/null | head -8)
    fi
  fi
else
  job_id="$1"
  shift || true
  log_glob=(
    "$ECHO_RUNTIME_ROOT"/logs/*-"$job_id".out
    "$ECHO_RUNTIME_ROOT"/logs/*-"$job_id".err
    "$REPO_ROOT"/logs/*-"$job_id".out
    "$REPO_ROOT"/logs/*-"$job_id".err
  )
fi

grep -Eh \
  "RUN_KIND=|RUN_ID=|CONFIG_PATH=|OUTPUT_DIR=|ENTRYPOINT_START=|ENTRYPOINT_DONE=|Started:|Finished:|Step [0-9]+:|global_step=|avg_final_rewards|avg_pass_at_|reward/|pass_at|save_checkpoint|save_checkpoints|latest_ckpt|policy_train|trainer_input|trainer_done|forward_backward_|optim_step_|policy_loss|final_loss|grad_norm|world_loss|world_ce|world_tokens|nextlat_|verifier_done|Traceback|RuntimeError|OutOfMemory|CUDA out of memory|FAILED|ModuleNotFound|FileNotFound" \
  "${log_glob[@]}" "$@" 2>/dev/null | tail -200
