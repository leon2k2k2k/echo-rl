#!/usr/bin/env bash
set -euo pipefail

# Install the remaining SkyRL/ECHO runtime wheels that are hard to resolve on
# Alex's cluster package mirrors. Keep this pinned and no-deps to avoid Torch drift.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/cluster_env.sh
source "$REPO_ROOT/scripts/cluster_env.sh"

if [[ ! -x "$ECHO_VENV/bin/python" ]]; then
  echo "error: ECHO_VENV does not contain python: $ECHO_VENV" >&2
  exit 1
fi

PY="$ECHO_VENV/bin/python"

echo "Using python: $PY"
"$PY" --version

echo "Checking Torch/TorchVision pair..."
if ! "$PY" - <<'PY'
import sys
import torch
import torchvision

ok = (
    torch.__version__.startswith("2.10.0")
    and torch.version.cuda == "12.8"
    and torchvision.__version__.startswith("0.25.0")
)
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("torchvision", torchvision.__version__)
sys.exit(0 if ok else 1)
PY
then
  echo "Repairing Torch/TorchVision/Triton to the cu128 set..."
  "$PY" -m pip install --force-reinstall --no-deps \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "triton==3.6.0"
fi

export PIP_NO_CACHE_DIR="${PIP_NO_CACHE_DIR:-1}"

echo "Installing pinned direct wheels with --no-deps..."
"$PY" -m pip install --no-deps \
  "vllm-router @ https://files.pythonhosted.org/packages/a8/b8/9ce36a665b11765c33f9ac8d47665d782dab637c140f79e5c2c865b00b6f/vllm_router-0.1.12-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "vllm @ https://files.pythonhosted.org/packages/b7/08/6a431731e4c163bc1fab85b63e269d84104aad0fba98dac1af34fdc5077f/vllm-0.19.0-cp38-abi3-manylinux_2_31_x86_64.whl" \
  "flash-linear-attention @ https://files.pythonhosted.org/packages/60/ee/a3cba17965482b35c4990af90bad108e82c32edcb59911c37f318b5f4198/flash_linear_attention-0.4.2-py3-none-any.whl" \
  "nixl @ https://files.pythonhosted.org/packages/7b/f8/e5bf11e31bcd42a86d6e5d0b7d166860d73d792ec6e7afbd7f04cf7eca06/nixl-1.0.0-py3-none-any.whl" \
  "nixl-cu12 @ https://files.pythonhosted.org/packages/48/68/f58b0b1aa8d2d03dd8354f6893fa858c77267ddef6cdd2d20868cf0ea88b/nixl_cu12-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl" \
  "causal-conv1d @ https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.1.post4/causal_conv1d-1.6.1%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

echo "Installing flashinfer-jit-cache separately because GitHub release downloads are flaky..."
"$PY" -m pip install --no-deps --timeout 120 --retries 10 \
  "flashinfer-jit-cache @ https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.6/flashinfer_jit_cache-0.6.6%2Bcu128-cp39-abi3-manylinux_2_28_x86_64.whl"

echo "Ensuring flashinfer-python is present without dependency resolution..."
"$PY" -m pip install --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "flashinfer-python==0.6.6"

echo "Installing lightweight vLLM runtime deps exposed by the import preflight..."
"$PY" -m pip install \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  "setproctitle" \
  "msgspec"

echo "Runtime wheel install finished. Run:"
echo "  $PY $REPO_ROOT/scripts/check_echo_runtime_imports.py"
