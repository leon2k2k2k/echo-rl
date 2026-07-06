#!/usr/bin/env bash
set -euo pipefail

# Source from the echo-rl repo before launching Alex-cluster jobs.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

DEFAULT_CLUSTER_CONFIG="$REPO_ROOT/configs/alex_cluster.env"
if [[ ! -f "$DEFAULT_CLUSTER_CONFIG" && -f "$REPO_ROOT/configs/alex_cluster.env.example" ]]; then
  DEFAULT_CLUSTER_CONFIG="$REPO_ROOT/configs/alex_cluster.env.example"
fi
CLUSTER_CONFIG="${ECHO_CLUSTER_CONFIG:-$DEFAULT_CLUSTER_CONFIG}"
if [[ -f "$CLUSTER_CONFIG" ]]; then
  # shellcheck source=/dev/null
  source "$CLUSTER_CONFIG"
fi

export ECHO_REPO="${ECHO_REPO:-$REPO_ROOT}"
export SKYRL_DIR="${SKYRL_DIR:-$(cd "$REPO_ROOT/../SkyRL" 2>/dev/null && pwd || true)}"
export ECHO_RUNTIME_ROOT="${ECHO_RUNTIME_ROOT:-$ECHO_REPO/runtime}"
export ECHO_VENV="${ECHO_VENV:-$ECHO_RUNTIME_ROOT/venv}"

export HF_HOME="${HF_HOME:-$ECHO_RUNTIME_ROOT/hf}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/hub}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$ECHO_RUNTIME_ROOT" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
mkdir -p "$ECHO_RUNTIME_ROOT/logs" "$ECHO_RUNTIME_ROOT/outputs" "$ECHO_RUNTIME_ROOT/results" "$ECHO_RUNTIME_ROOT/tmp"

if [[ -x "$ECHO_VENV/bin/python" && ( -z "${PYTHON:-}" || "${PYTHON:-}" == "python" || "${PYTHON:-}" == "python3" ) ]]; then
  export PYTHON="$ECHO_VENV/bin/python"
fi
export PYTHON="${PYTHON:-python3}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
