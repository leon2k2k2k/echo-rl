#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/cluster_env.sh"

shopt -s nullglob
logs=(
  "$ECHO_RUNTIME_ROOT"/logs/echo-rl-smoke-*.out
  "$ECHO_RUNTIME_ROOT"/logs/echo-rl-smoke-*.err
  "$REPO_ROOT"/logs/echo-rl-smoke-*.out
  "$REPO_ROOT"/logs/echo-rl-smoke-*.err
)

if (( ${#logs[@]} == 0 )); then
  echo "No echo-rl-smoke logs found yet under:"
  echo "  $ECHO_RUNTIME_ROOT/logs"
  echo "  $REPO_ROOT/logs"
  exit 1
fi

printf 'Tailing %d echo smoke log files:\n' "${#logs[@]}"
printf '  %s\n' "${logs[@]}"
tail -f "${logs[@]}"
