#!/usr/bin/env python3
"""Summarize terminal-agent pass@16 scores from Slurm logs."""

from __future__ import annotations

import argparse
import ast
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"Step\s+(\d+):")
PREFIX_RE = re.compile(r"^\([^)]*\)\s*")
ASSIGN_RE = re.compile(r"\b(RUN_KIND|RUN_ID|CONFIG_PATH|OUTPUT_DIR|TERMINAL_AGENT_TRAIN_PARQUET)=(.*)$")
WARNING_RE = re.compile(r"\bWARNING\b.*?\s-\s(.*)$")
TIMESTAMP_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:[,.]\d+)?")
TOTAL_STEPS_RE = re.compile(r"Total steps:\s*(\d+)")
PHASE_RE = re.compile(r"\b(Started|Finished): '([^']+)'")
ROLLOUT_RE = re.compile(r"rollout_done .*?sec=([0-9.]+).*?reward=([0-9.]+)")


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


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _parse_timestamp(line: str) -> datetime | None:
    match = TIMESTAMP_RE.search(line)
    if not match:
        return None
    text = match.group(1).replace("T", " ")
    try:
        return datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _extract_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for raw in lines:
        line = _clean_line(raw)
        match = ASSIGN_RE.search(line)
        if match:
            metadata[match.group(1)] = match.group(2).strip()
    return metadata


def _extract_warnings(lines: list[str]) -> Counter[str]:
    warnings: Counter[str] = Counter()
    for raw in lines:
        if "WARNING" not in raw:
            continue
        line = _clean_line(raw)
        match = WARNING_RE.search(line)
        message = match.group(1).strip() if match else line
        warnings[message] += 1
    return warnings


def _extract_progress(lines: list[str], steps: list[dict[str, Any]]) -> dict[str, Any]:
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    total_steps: int | None = None
    last_phase: str | None = None
    active_phase: str | None = None
    rollout_secs: list[float] = []
    last_rollout_reward: float | None = None

    for raw in lines:
        line = _clean_line(raw)
        timestamp = _parse_timestamp(line)
        if timestamp is not None:
            first_ts = first_ts or timestamp
            last_ts = timestamp

        match = TOTAL_STEPS_RE.search(line)
        if match:
            total_steps = int(match.group(1))

        match = PHASE_RE.search(line)
        if match:
            action, phase = match.groups()
            last_phase = f"{action.lower()}:{phase}"
            active_phase = phase if action == "Started" else None

        match = ROLLOUT_RE.search(line)
        if match:
            rollout_secs.append(float(match.group(1)))
            last_rollout_reward = float(match.group(2))

    completed_steps = len(steps)
    latest_step = steps[-1].get("_step") if steps else None
    step_times = [_to_float(step.get("timing/step")) for step in steps]
    step_times = [value for value in step_times if value is not None and value > 0]
    recent_step_times = step_times[-5:]
    avg_step_sec = sum(recent_step_times) / len(recent_step_times) if recent_step_times else None
    remaining_steps = None
    eta_sec = None
    if total_steps is not None:
        # The tracking step number can be implementation-dependent, so use
        # parsed metric dictionaries as the conservative completed count.
        remaining_steps = max(0, total_steps - completed_steps)
        if avg_step_sec is not None:
            eta_sec = remaining_steps * avg_step_sec

    elapsed_sec = (last_ts - first_ts).total_seconds() if first_ts and last_ts else None
    avg_rollout_sec = sum(rollout_secs) / len(rollout_secs) if rollout_secs else None

    return {
        "first_ts": first_ts,
        "last_ts": last_ts,
        "elapsed_sec": elapsed_sec,
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "latest_step": latest_step,
        "remaining_steps": remaining_steps,
        "avg_step_sec": avg_step_sec,
        "eta_sec": eta_sec,
        "last_phase": last_phase,
        "active_phase": active_phase,
        "rollout_count": len(rollout_secs),
        "avg_rollout_sec": avg_rollout_sec,
        "last_rollout_reward": last_rollout_reward,
    }


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
    warnings = _extract_warnings(lines)
    progress = _extract_progress(lines, steps)
    verifier_timeout_lines = sum("verifier_timeout" in line for line in lines)
    verifier_error_lines = sum("verifier_error" in line for line in lines)
    rollout_reward_one = sum("rollout_done" in line and "reward=1.0" in line for line in lines)
    rollout_reward_zero = sum("rollout_done" in line and "reward=0.0" in line for line in lines)
    latest = steps[-1] if steps else {}

    return {
        "paths": existing,
        "metadata": metadata,
        "warnings": warnings,
        "progress": progress,
        "steps": steps,
        "latest": latest,
        "verifier_timeout_lines": verifier_timeout_lines,
        "verifier_error_lines": verifier_error_lines,
        "rollout_reward_one": rollout_reward_one,
        "rollout_reward_zero": rollout_reward_zero,
    }


def _print_summary(label: str, summary: dict[str, Any], show_steps: bool, warning_limit: int) -> None:
    metadata = summary["metadata"]
    latest = summary["latest"]
    steps = summary["steps"]
    progress = summary["progress"]

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
    if progress["first_ts"] or progress["last_ts"] or progress["total_steps"] is not None:
        first_ts = progress["first_ts"].strftime("%H:%M:%S") if progress["first_ts"] else "-"
        last_ts = progress["last_ts"].strftime("%H:%M:%S") if progress["last_ts"] else "-"
        total_steps = progress["total_steps"] if progress["total_steps"] is not None else "?"
        completed_steps = progress["completed_steps"]
        remaining_steps = progress["remaining_steps"] if progress["remaining_steps"] is not None else "?"
        print(
            "progress: "
            f"{completed_steps}/{total_steps} metric_steps "
            f"remaining={remaining_steps} "
            f"elapsed={_fmt_duration(progress['elapsed_sec'])} "
            f"eta={_fmt_duration(progress['eta_sec'])} "
            f"avg_step={_fmt(progress['avg_step_sec'], 1)}s "
            f"log_window={first_ts}->{last_ts}"
        )
        print(
            "phase:    "
            f"active={progress['active_phase'] or '-'} "
            f"last={progress['last_phase'] or '-'} "
            f"rollouts={progress['rollout_count']} "
            f"avg_rollout={_fmt(progress['avg_rollout_sec'], 1)}s "
            f"last_rollout_reward={_fmt(progress['last_rollout_reward'], 1)}"
        )
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
    warnings = summary["warnings"]
    print(f"warnings: {sum(warnings.values())}")
    if warning_limit > 0 and warnings:
        for message, count in warnings.most_common(warning_limit):
            print(f"  [{count}] {message}")

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
    parser.add_argument("--warnings", type=int, default=5, help="Number of warning messages to print per job. Default: 5.")
    args = parser.parse_args()

    for item in args.jobs_or_logs:
        paths = _log_paths_for_arg(item)
        summary = _summarize_logs(paths)
        _print_summary(item, summary, show_steps=args.steps, warning_limit=args.warnings)


if __name__ == "__main__":
    main()
