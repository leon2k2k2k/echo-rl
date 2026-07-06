#!/usr/bin/env python3
"""Fail fast if the ECHO/SkyRL runtime environment is missing imports."""

from __future__ import annotations

import importlib
import sys


REQUIRED_IMPORTS = [
    # ECHO direct/runtime deps.
    ("datasets", "datasets"),
    ("harbor", "harbor"),
    ("loguru", "loguru"),
    ("omegaconf", "omegaconf"),
    ("ray", "ray"),
    ("torch", "torch"),
    ("transformers", "transformers"),
    # SkyRL base/train deps reached by the terminal-agent path.
    ("accelerate", "accelerate"),
    ("cloudpathlib", "cloudpathlib"),
    ("func_timeout", "func_timeout"),
    ("hydra", "hydra"),
    ("jaxtyping", "jaxtyping"),
    ("peft", "peft"),
    ("safetensors", "safetensors"),
    ("tensorboard", "tensorboard"),
    ("tensordict", "tensordict"),
    ("torchdata", "torchdata"),
    ("torchdata.stateful_dataloader", "torchdata.stateful_dataloader"),
    # SkyRL local package and gym extra.
    ("skyrl", "skyrl"),
    ("skyrl_gym", "skyrl_gym"),
    # FSDP + vLLM inference path.
    ("vllm", "vllm"),
    ("vllm_router", "vllm-router"),
    ("flash_attn", "flash-attn"),
    ("flash_attn.bert_padding", "flash-attn"),
    ("nixl", "nixl"),
    ("causal_conv1d", "causal-conv1d"),
    ("fla", "flash-linear-attention"),
    ("flashinfer", "flashinfer-python"),
]


def main() -> int:
    failed = []
    for module, package in REQUIRED_IMPORTS:
        try:
            imported = importlib.import_module(module)
            version = getattr(imported, "__version__", "")
            print(f"OK   {module}" + (f" {version}" if version else ""))
        except Exception as exc:
            print(f"FAIL {module} ({package}): {type(exc).__name__}: {exc}")
            failed.append((module, package))

    if failed:
        print("\nMissing runtime imports. Install/fix these before submitting Slurm jobs:")
        for module, package in failed:
            print(f"  module={module} package={package}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
