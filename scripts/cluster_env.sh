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
mkdir -p "$ECHO_RUNTIME_ROOT/data"

export ECHO_SMOKE_DATA_DIR="${ECHO_SMOKE_DATA_DIR:-$ECHO_RUNTIME_ROOT/data}"
export ECHO_DEFAULT_TRAIN_PARQUET="$ECHO_SMOKE_DATA_DIR/terminal_agent_smoke_train.parquet"
export ECHO_DEFAULT_VAL_PARQUET="$ECHO_SMOKE_DATA_DIR/terminal_agent_smoke_val.parquet"
export TERMINAL_AGENT_TRAIN_PARQUET="${TERMINAL_AGENT_TRAIN_PARQUET:-$ECHO_DEFAULT_TRAIN_PARQUET}"
export TERMINAL_AGENT_VAL_PARQUET="${TERMINAL_AGENT_VAL_PARQUET:-$ECHO_DEFAULT_VAL_PARQUET}"

if [[ -x "$ECHO_VENV/bin/python" && ( -z "${PYTHON:-}" || "${PYTHON:-}" == "python" || "${PYTHON:-}" == "python3" ) ]]; then
  export PYTHON="$ECHO_VENV/bin/python"
fi
export PYTHON="${PYTHON:-python3}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

ensure_terminal_agent_smoke_data() {
  if [[ "$TERMINAL_AGENT_TRAIN_PARQUET" != "$ECHO_DEFAULT_TRAIN_PARQUET" ||
        "$TERMINAL_AGENT_VAL_PARQUET" != "$ECHO_DEFAULT_VAL_PARQUET" ]]; then
    return 0
  fi

  if [[ -f "$TERMINAL_AGENT_TRAIN_PARQUET" && -f "$TERMINAL_AGENT_VAL_PARQUET" ]]; then
    return 0
  fi

  "$PYTHON" "$ECHO_REPO/scripts/create_terminal_agent_smoke_data.py" \
    --train "$TERMINAL_AGENT_TRAIN_PARQUET" \
    --val "$TERMINAL_AGENT_VAL_PARQUET" \
    --rows 4
}

check_terminal_agent_data_files() {
  local missing=0
  local name value

  for name in TERMINAL_AGENT_TRAIN_PARQUET TERMINAL_AGENT_VAL_PARQUET; do
    value="${!name:-}"
    if [[ -z "$value" || ! -f "$value" ]]; then
      echo "error: $name does not point to an existing data file: ${value:-<unset>}" >&2
      missing=1
    fi
  done

  if (( missing )); then
    cat >&2 <<'EOF'

Find the terminal-agent parquet files on Alex, then export the paths before submitting:

  find /home/fit/alex/WORK /WORK/PUBLIC/alex_work /mnt -type f \
    \( -name 'train*sa_q35xml*.parquet' -o -name 'val*sa_q35xml*.parquet' \) \
    2>/dev/null | sort

  export TERMINAL_AGENT_TRAIN_PARQUET=/path/to/train.parquet
  export TERMINAL_AGENT_VAL_PARQUET=/path/to/val.parquet
  bash scripts/submit_echo_smokes.sh

Or put those exports in configs/alex_cluster.env for this checkout.
EOF
    return 1
  fi
}
