#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/cluster_env.sh"

if [[ -z "${SKYRL_DIR:-}" || ! -d "$SKYRL_DIR/.git" ]]; then
  echo "error: SKYRL_DIR must point to a SkyRL git checkout, got ${SKYRL_DIR:-<unset>}" >&2
  exit 1
fi

patch_file="$REPO_ROOT/patches/skyrl_minimal_hooks.patch"

if git -C "$SKYRL_DIR" apply --reverse --check "$patch_file" >/dev/null 2>&1; then
  echo "SkyRL minimal hooks already applied in $SKYRL_DIR"
elif [[ -z "$(git -C "$SKYRL_DIR" status --short)" ]] && git -C "$SKYRL_DIR" apply --check "$patch_file" >/dev/null 2>&1; then
  git -C "$SKYRL_DIR" apply "$patch_file"
  echo "Applied SkyRL minimal hooks in $SKYRL_DIR"
else
  echo "error: SkyRL checkout is neither clean-applicable nor already patched." >&2
  echo "       Resolve unrelated changes or set SKYRL_DIR to a prepared checkout." >&2
  git -C "$SKYRL_DIR" status --short >&2
  exit 1
fi

cp -a "$REPO_ROOT/echo_rl" "$SKYRL_DIR/"
mkdir -p "$SKYRL_DIR/echo_configs"
cp -a "$REPO_ROOT/configs/." "$SKYRL_DIR/echo_configs/"
cp "$REPO_ROOT/scripts/run_echo_terminal_agent.sh" "$SKYRL_DIR/run_echo_terminal_agent.sh"
chmod +x "$SKYRL_DIR/run_echo_terminal_agent.sh"

echo "Synced ECHO package, configs, and launcher into $SKYRL_DIR"
