#!/usr/bin/env python3
"""Prebuild and smoke-test terminal-agent Harbor Docker sandboxes.

This exercises the Docker environment path used by RL generation without
starting vLLM or running a policy update.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from echo_rl.terminal_agent.harbor_environment import HarborEnvironmentProvider


def _load_environment_rows(parquet_path: str, rows: int) -> list[dict[str, Any]]:
    table = pq.read_table(parquet_path, columns=["path", "task_binary"])
    output = []
    for row in table.slice(0, rows).to_pylist():
        task_binary = row.get("task_binary")
        if task_binary is None:
            raise ValueError(f"row {len(output)} is missing task_binary")
        output.append(
            {
                "path": row.get("path") or f"row_{len(output)}",
                "task_binary": bytes(task_binary),
            }
        )
    return output


async def _run(args: argparse.Namespace) -> int:
    os.environ.setdefault("DOCKER_BUILDKIT", "0")
    os.environ.setdefault("COMPOSE_DOCKER_CLI_BUILD", "0")
    os.environ.setdefault("ECHO_CACHE_TASK_IMAGES", "1")
    if "ECHO_DOCKER_WHEELHOUSE" not in os.environ:
        runtime_root = os.environ.get("ECHO_RUNTIME_ROOT")
        if runtime_root:
            os.environ["ECHO_DOCKER_WHEELHOUSE"] = str(Path(runtime_root) / "docker_wheelhouse")
    os.environ.setdefault("ECHO_DOCKER_PIP_INDEX_URL", "https://pypi.tuna.tsinghua.edu.cn/simple")

    rows = _load_environment_rows(args.parquet, args.rows)
    provider = HarborEnvironmentProvider(
        docker_memory_mb=args.docker_memory_mb,
        docker_cpus=args.docker_cpus,
        base_temp_dir=Path(args.base_temp_dir) if args.base_temp_dir else None,
        max_concurrent_builds=args.max_concurrent_builds,
        max_build_retries=args.max_build_retries,
    )

    print(f"parquet={args.parquet}")
    print(f"rows={len(rows)}")
    print(f"DOCKER_BUILDKIT={os.environ.get('DOCKER_BUILDKIT')}")
    print(f"COMPOSE_DOCKER_CLI_BUILD={os.environ.get('COMPOSE_DOCKER_CLI_BUILD')}")
    print(f"ECHO_CACHE_TASK_IMAGES={os.environ.get('ECHO_CACHE_TASK_IMAGES')}")
    print(f"ECHO_DOCKER_WHEELHOUSE={os.environ.get('ECHO_DOCKER_WHEELHOUSE')}")
    print(f"ECHO_DOCKER_PIP_INDEX_URL={os.environ.get('ECHO_DOCKER_PIP_INDEX_URL')}")
    print(f"max_concurrent_builds={args.max_concurrent_builds}")

    t0 = time.monotonic()
    await provider.prepare_batch(rows, num_generations=1)
    print(f"prepare_batch_sec={time.monotonic() - t0:.1f}")

    failed = 0
    for idx, row in enumerate(rows):
        env = None
        task = row["path"]
        print(f"\n=== row {idx}: {task} ===")
        try:
            env = await provider.create(row)
            t_setup = time.monotonic()
            await env.setup()
            print(f"setup_sec={time.monotonic() - t_setup:.1f}")

            result = await env.exec(args.command, timeout=args.command_timeout)
            stdout = (result.stdout or "").strip()
            stderr = (result.stderr or "").strip()
            print(f"command_return_code={result.return_code}")
            if stdout:
                print("stdout:")
                print(stdout[: args.output_chars])
            if stderr:
                print("stderr:")
                print(stderr[: args.output_chars])
            if result.return_code != 0:
                failed += 1

            if args.run_verifier:
                t_verify = time.monotonic()
                reward, reason = await env.run_verifier(timeout=args.verifier_timeout)
                print(
                    f"verifier_reward={reward} verifier_reason={reason} "
                    f"verifier_sec={time.monotonic() - t_verify:.1f}"
                )
        except Exception as exc:
            failed += 1
            print(f"FAILED row {idx}: {type(exc).__name__}: {exc}")
            logging.exception("row %s failed", idx)
        finally:
            if env is not None:
                await env.cleanup()

    if args.cleanup_shared:
        await provider.cleanup_batch()
    else:
        print("\nLeaving built hb__ task images and shared build dirs in place.")
        print("Use --cleanup-shared if you want this script to remove them.")

    if failed:
        print(f"\nFAILED rows={failed}/{len(rows)}")
        return 1
    print(f"\nOK rows={len(rows)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--base-temp-dir", default=os.environ.get("ECHO_TBENCH_PREFLIGHT_DIR"))
    parser.add_argument("--docker-memory-mb", type=int, default=1024)
    parser.add_argument("--docker-cpus", type=int, default=1)
    parser.add_argument("--max-concurrent-builds", type=int, default=1)
    parser.add_argument("--max-build-retries", type=int, default=1)
    parser.add_argument(
        "--command",
        default="pwd && ls -la && test -f /tests/test.sh && echo TEST_SH_PRESENT",
    )
    parser.add_argument("--command-timeout", type=float, default=60.0)
    parser.add_argument("--output-chars", type=int, default=2000)
    parser.add_argument("--run-verifier", action="store_true")
    parser.add_argument("--verifier-timeout", type=float, default=120.0)
    parser.add_argument("--cleanup-shared", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
