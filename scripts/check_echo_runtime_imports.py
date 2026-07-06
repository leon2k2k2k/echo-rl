#!/usr/bin/env python3
"""Fail fast if the ECHO/SkyRL runtime environment is missing imports."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import sys


REQUIRED_DISTS = [
    # ECHO pyproject deps.
    "datasets",
    "harbor",
    "loguru",
    "omegaconf",
    "ray",
    "torch",
    "tqdm",
    "transformers",
    "typing-extensions",
    "pyarrow",
    # SkyRL base deps.
    "pillow",
    "rich",
    "safetensors",
    "tokenizers",
    "typer",
    "peft",
    "hf-transfer",
    "cloudpathlib",
    # SkyRL skyrl-train deps.
    "ninja",
    "tensorboard",
    "func-timeout",
    "hydra-core",
    "accelerate",
    "torchdata",
    "debugpy",
    "wandb",
    "tensordict",
    "jaxtyping",
    "skyrl-gym",
    "flash-attn",
    "polars",
    "s3fs",
    "fastapi",
    "uvicorn",
    "vllm-router",
    "gguf",
    "setproctitle",
    "msgspec",
    "uvloop",
    "pybind11",
    "nixl",
    # SkyRL fsdp deps.
    "vllm",
    "flash-linear-attention",
    "causal-conv1d",
    "flashinfer-python",
    "flashinfer-jit-cache",
    "torchvision",
]

FAST_IMPORTS = [
    # ECHO direct/runtime deps.
    ("datasets", "datasets"),
    ("harbor", "harbor"),
    ("loguru", "loguru"),
    ("omegaconf", "omegaconf"),
    ("ray", "ray"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("pyarrow", "pyarrow"),
    ("pyarrow.parquet", "pyarrow"),
    ("PIL", "pillow"),
    ("rich", "rich"),
    ("tokenizers", "tokenizers"),
    ("typer", "typer"),
    ("tqdm", "tqdm"),
    # SkyRL base/train deps reached by the terminal-agent path.
    ("accelerate", "accelerate"),
    ("cloudpathlib", "cloudpathlib"),
    ("debugpy", "debugpy"),
    ("func_timeout", "func_timeout"),
    ("hydra", "hydra"),
    ("hf_transfer", "hf-transfer"),
    ("jaxtyping", "jaxtyping"),
    ("ninja", "ninja"),
    ("peft", "peft"),
    ("polars", "polars"),
    ("safetensors", "safetensors"),
    ("s3fs", "s3fs"),
    ("tensorboard", "tensorboard"),
    ("tensordict", "tensordict"),
    ("torchdata", "torchdata"),
    ("torchdata.stateful_dataloader", "torchdata.stateful_dataloader"),
    ("wandb", "wandb"),
    # SkyRL local package and gym extra.
    ("skyrl", "skyrl"),
    ("skyrl_gym", "skyrl_gym"),
    ("echo_rl.terminal_agent.entrypoint", "echo-rl"),
    ("echo_rl.world_modeling.fsdp_worker", "echo-rl"),
]

DEEP_IMPORTS = [
    # FSDP + vLLM inference path.
    ("vllm", "vllm"),
    ("vllm_router", "vllm-router"),
    ("gguf", "gguf"),
    ("setproctitle", "setproctitle"),
    ("msgspec", "msgspec"),
    ("uvloop", "uvloop"),
    ("flash_attn", "flash-attn"),
    ("flash_attn.bert_padding", "flash-attn"),
    ("nixl", "nixl"),
    ("causal_conv1d", "causal-conv1d"),
    ("fla", "flash-linear-attention"),
    ("flashinfer", "flashinfer-python"),
    # Exact SkyRL/ECHO module paths reached before training starts.
    ("skyrl.train.trainer", "skyrl"),
    ("skyrl.train.evaluate", "skyrl"),
    ("skyrl.backends.skyrl_train.inference_servers.setup", "skyrl"),
    ("skyrl.backends.skyrl_train.inference_engines.vllm.vllm_engine", "skyrl"),
    ("skyrl.backends.skyrl_train.workers.fsdp.fsdp_worker", "skyrl"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check the ECHO/SkyRL runtime environment."
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="also import slow CUDA/vLLM/SkyRL launch-path modules",
    )
    args = parser.parse_args()

    failed = []
    print("Checking installed distributions:")
    for dist in REQUIRED_DISTS:
        try:
            version = importlib.metadata.version(dist)
            print(f"OK   dist {dist} {version}")
        except importlib.metadata.PackageNotFoundError:
            print(f"FAIL dist {dist}: not installed")
            failed.append((f"dist:{dist}", dist))

    imports = FAST_IMPORTS + (DEEP_IMPORTS if args.deep else [])
    mode = "deep" if args.deep else "fast"
    print(f"\nChecking {mode} imports:")
    for module, package in imports:
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
    if not args.deep:
        print("\nFast check passed. Run with --deep only after env changes or import failures.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
