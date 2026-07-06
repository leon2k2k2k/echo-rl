#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/cluster_env.sh"

shopt -s nullglob

logs=()

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
      [[ -f "$log" ]] && logs+=("$log")
    done
  done
fi

if (( ${#logs[@]} == 0 )); then
  candidates=(
    "$ECHO_RUNTIME_ROOT"/logs/echo-rl-smoke-*.out
    "$ECHO_RUNTIME_ROOT"/logs/echo-rl-smoke-*.err
    "$REPO_ROOT"/logs/echo-rl-smoke-*.out
    "$REPO_ROOT"/logs/echo-rl-smoke-*.err
  )

  if (( ${#candidates[@]} > 0 )); then
    mapfile -t logs < <(ls -t "${candidates[@]}" 2>/dev/null | head -8)
  fi
fi

if (( ${#logs[@]} == 0 )); then
  echo "No echo-rl-smoke logs found yet under:"
  echo "  $ECHO_RUNTIME_ROOT/logs"
  echo "  $REPO_ROOT/logs"
  exit 1
fi

printf 'Tailing %d echo smoke log files:\n' "${#logs[@]}"
printf '  %s\n' "${logs[@]}"
tail -f "${logs[@]}"
