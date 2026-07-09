#!/usr/bin/env python3
"""Summarize terminal-agent pass@16 scores from Slurm logs."""

from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"Step\s+(\d+):")
PREFIX_RE = re.compile(r"^\([^)]*\)\s*")
ASSIGN_RE = re.compile(r"\b(RUN_KIND|RUN_ID|CONFIG_PATH|OUTPUT_DIR|TERMINAL_AGENT_TRAIN_PARQUET)=(.*)$")


def _runtime_root() -> Path:
    return Path(os.environ.get("ECHO_RUNTIME_ROOT", "/home/fit/alex/WORK/leon/echo_rl_runtime"))


def _log_paths_for_arg(arg: str) -> list[Path]:
    path = Path(arg)
    if path.exists():
        return [path]
    if arg.isdigit():
        root = _runtime_root()
        return [
            root / "logs" / f"echo-rl-smoke-{arg}.out",
            root / "logs" / f"echo-rl-smoke-{arg}.err",
        ]
    return [path]


def _clean_line(line: str) -> str:
    return PREFIX_RE.sub("", line).strip()


def _brace_delta(text: str) -> int:
    in_single = False
    in_double = False
    escaped = False
    delta = 0
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if char == "{":
            delta += 1
        elif char == "}":
            delta -= 1
    return delta


def _parse_metric_dicts(lines: list[str]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    pending_step: int | None = None
    collecting = False
    brace_balance = 0
    buf: list[str] = []

    for raw in lines:
        line = _clean_line(raw)
        match = STEP_RE.search(line)
        if match:
            pending_step = int(match.group(1))
            collecting = False
            brace_balance = 0
            buf = []
            continue

        if pending_step is None:
            continue

        if not collecting:
            brace_idx = line.find("{")
            if brace_idx < 0:
                continue
            line = line[brace_idx:]
            collecting = True

        buf.append(line)
        brace_balance += _brace_delta(line)
        if collecting and brace_balance <= 0 and "}" in line:
            text = "\n".join(buf)
            try:
                metrics = ast.literal_eval(text)
            except Exception:
                pending_step = None
                collecting = False
                buf = []
                continue
            if isinstance(metrics, dict):
                metrics["_step"] = pending_step
                steps.append(metrics)
            pending_step = None
            collecting = False
            buf = []
            brace_balance = 0

    return steps


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _fmt(value: Any, digits: int = 3) -> str:
    number = _to_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _extract_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in lines:
        line = _clean_line(raw)
        match = ASSIGN_RE.search(line)
        if match:
            metadata[match.group(1)] = match.group(2).strip()
    return metadata


def _summarize_logs(paths: list[Path]) -> dict[str, Any]:
    lines: list[str] = []
    existing = [path for path in paths if path.exists()]
    for path in existing:
        try:
            lines.extend(path.read_text(errors="replace").splitlines())
        except OSError:
            pass

    steps = _parse_metric_dicts(lines)
    metadata = _extract_metadata(lines)
    verifier_timeout_lines = sum("verifier_timeout" in line for line in lines)
    verifier_error_lines = sum("verifier_error" in line for line in lines)
    rollout_reward_one = sum("rollout_done" in line and "reward=1.0" in line for line in lines)
    rollout_reward_zero = sum("rollout_done" in line and "reward=0.0" in line for line in lines)
    latest = steps[-1] if steps else {}

    return {
        "paths": existing,
        "metadata": metadata,
        "steps": steps,
        "latest": latest,
        "verifier_timeout_lines": verifier_timeout_lines,
        "verifier_error_lines": verifier_error_lines,
        "rollout_reward_one": rollout_reward_one,
        "rollout_reward_zero": rollout_reward_zero,
    }


def _print_summary(label: str, summary: dict[str, Any], show_steps: bool) -> None:
    metadata = summary["metadata"]
    latest = summary["latest"]
    steps = summary["steps"]

    print(f"\n== {label} ==")
    if metadata.get("RUN_KIND"):
        print(f"run_kind: {metadata['RUN_KIND']}")
    if metadata.get("CONFIG_PATH"):
        print(f"config:   {metadata['CONFIG_PATH']}")
    if metadata.get("TERMINAL_AGENT_TRAIN_PARQUET"):
        print(f"train:    {metadata['TERMINAL_AGENT_TRAIN_PARQUET']}")
    if not summary["paths"]:
        print("logs:     none found")
        return

    print(f"logs:     {', '.join(str(path) for path in summary['paths'])}")
    print(f"steps:    {len(steps)}")
    if latest:
        print(
            "latest:   "
            f"step={latest.get('_step')} "
            f"pass16={_fmt(latest.get('reward/avg_pass_at_16'))} "
            f"raw={_fmt(latest.get('reward/avg_raw_reward'))} "
            f"loss_reward={_fmt(latest.get('loss/avg_final_rewards'))} "
            f"time_step={_fmt(latest.get('timing/step'), 1)}s "
            f"time_gen={_fmt(latest.get('timing/generate'), 1)}s"
        )
        print(
            "latest_counts: "
            f"done={_to_int(latest.get('generate/stop_reason/done'))} "
            f"max_turns={_to_int(latest.get('generate/stop_reason/max_turns'))} "
            f"max_tokens={_to_int(latest.get('generate/stop_reason/max_total_tokens'))} "
            f"verifier_timeout={_to_int(latest.get('generate/verifier_error/verifier_timeout'))} "
            f"verifier_error={_to_int(latest.get('generate/verifier_error/verifier_error'))} "
            f"parse_errors={_to_int(latest.get('generate/parse_errors'))} "
            f"format_violations={_to_int(latest.get('generate/format_violations_total'))}"
        )
    print(
        "rollouts_seen: "
        f"reward1={summary['rollout_reward_one']} "
        f"reward0={summary['rollout_reward_zero']} "
        f"verifier_timeout_lines={summary['verifier_timeout_lines']} "
        f"verifier_error_lines={summary['verifier_error_lines']}"
    )

    if show_steps and steps:
        print("\nstep pass16 raw loss_reward timeout verr parse fmt step_s gen_s")
        for metrics in steps:
            print(
                f"{metrics.get('_step'):>4} "
                f"{_fmt(metrics.get('reward/avg_pass_at_16')):>6} "
                f"{_fmt(metrics.get('reward/avg_raw_reward')):>6} "
                f"{_fmt(metrics.get('loss/avg_final_rewards')):>11} "
                f"{_to_int(metrics.get('generate/verifier_error/verifier_timeout')):>7} "
                f"{_to_int(metrics.get('generate/verifier_error/verifier_error')):>4} "
                f"{_to_int(metrics.get('generate/parse_errors')):>5} "
                f"{_to_int(metrics.get('generate/format_violations_total')):>3} "
                f"{_fmt(metrics.get('timing/step'), 1):>6} "
                f"{_fmt(metrics.get('timing/generate'), 1):>5}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs_or_logs", nargs="+", help="Slurm job IDs or log file paths.")
    parser.add_argument("--steps", action="store_true", help="Print per-step table.")
    args = parser.parse_args()

    for item in args.jobs_or_logs:
        paths = _log_paths_for_arg(item)
        summary = _summarize_logs(paths)
        _print_summary(item, summary, show_steps=args.steps)


if __name__ == "__main__":
    main()
