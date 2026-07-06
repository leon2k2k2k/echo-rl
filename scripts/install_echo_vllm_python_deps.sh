#!/usr/bin/env bash
set -euo pipefail

# Install vLLM's non-Torch Python/runtime deps after vLLM itself has been
# installed with --no-deps. This avoids resolver-driven Torch/CUDA changes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/cluster_env.sh
source "$REPO_ROOT/scripts/cluster_env.sh"

if [[ ! -x "$ECHO_VENV/bin/python" ]]; then
  echo "error: ECHO_VENV does not contain python: $ECHO_VENV" >&2
  exit 1
fi

PY="$ECHO_VENV/bin/python"

"$PY" -m pip install --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "anthropic" \
  "blake3" \
  "cachetools" \
  "cbor2" \
  "cloudpickle" \
  "compressed-tensors" \
  "depyf" \
  "diskcache" \
  "einops" \
  "gguf" \
  "ijson" \
  "lark" \
  "llguidance" \
  "lm-format-enforcer" \
  "mistral-common" \
  "model-hosting-container-standards" \
  "numba" \
  "openai-harmony" \
  "opencv-python-headless" \
  "outlines-core" \
  "partial-json-parser" \
  "prometheus-client" \
  "prometheus-fastapi-instrumentator" \
  "py-cpuinfo" \
  "pybase64" \
  "python-json-logger" \
  "pyzmq" \
  "sentencepiece" \
  "setproctitle" \
  "msgspec" \
  "watchfiles" \
  "xgrammar" \
  "orjson"

echo "vLLM Python deps installed. Run:"
echo "  $PY $REPO_ROOT/scripts/check_echo_runtime_imports.py"
