#!/usr/bin/env python3
"""Pretty-print terminal-agent transcript JSONL files."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable


def _open_follow(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8") as f:
        while True:
            line = f.readline()
            if line:
                yield line
                continue
            time.sleep(0.5)


def _iter_lines(path: Path, follow: bool) -> Iterable[str]:
    if follow:
        yield from _open_follow(path)
    else:
        with path.open("r", encoding="utf-8") as f:
            yield from f


def _print_messages(messages: list[dict[str, Any]], max_chars: int) -> None:
    for message in messages:
        content = str(message.get("content", ""))
        if max_chars > 0 and len(content) > max_chars:
            content = content[:max_chars] + f"\n...[truncated {len(content) - max_chars} chars]"
        print(f"\n[{message.get('role', '?')}]\n{content}")


def _print_event(event: dict[str, Any], args: argparse.Namespace) -> None:
    event_name = event.get("event", "?")
    trajectory = event.get("trajectory_id", "?")
    turn = event.get("turn", "-")
    path = event.get("path", "")
    print(f"\n=== {event_name} {trajectory} turn={turn} {path} ===")

    if event_name == "final":
        print(
            "reward={reward} correct={correct} stop_reason={stop_reason}".format(
                reward=event.get("reward"),
                correct=event.get("correct"),
                stop_reason=event.get("stop_reason"),
            )
        )
        trace = event.get("trace") or {}
        if trace:
            print(
                "agent_sec={agent:.2f} generate_sec={gen:.2f} exec_sec={exec:.2f} "
                "verifier_sec={verifier:.2f} tokens={tokens}".format(
                    agent=float(trace.get("agent_run_sec") or 0.0),
                    gen=float(trace.get("total_generate_sec") or 0.0),
                    exec=float(trace.get("total_exec_sec") or 0.0),
                    verifier=float(trace.get("verifier_sec") or 0.0),
                    tokens=trace.get("total_tokens"),
                )
            )
        if args.full_final_messages and event.get("messages"):
            _print_messages(event["messages"], args.max_chars)
        return

    content = event.get("content")
    if isinstance(content, str):
        if args.max_chars > 0 and len(content) > args.max_chars:
            content = content[: args.max_chars] + f"\n...[truncated {len(content) - args.max_chars} chars]"
        print(content)
    elif event.get("messages"):
        _print_messages(event["messages"], args.max_chars)

    extras = []
    for key in ("generate_sec", "generate_tokens", "exec_sec", "parse_error", "format_violations", "commands"):
        if key in event:
            extras.append(f"{key}={event[key]}")
    if extras:
        print("\n" + " ".join(extras))


def _print_final_summary(path: Path) -> int:
    count = 0
    rewards: list[float] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "final":
                continue
            count += 1
            reward = float(event.get("reward") or 0.0)
            rewards.append(reward)
            trace = event.get("trace") or {}
            print(
                "{trajectory} {path} reward={reward} correct={correct} stop={stop} "
                "turns={turns} agent_sec={agent:.2f} gen_sec={gen:.2f} exec_sec={exec:.2f}".format(
                    trajectory=event.get("trajectory_id"),
                    path=event.get("path"),
                    reward=event.get("reward"),
                    correct=event.get("correct"),
                    stop=event.get("stop_reason"),
                    turns=len(trace.get("turns") or []),
                    agent=float(trace.get("agent_run_sec") or 0.0),
                    gen=float(trace.get("total_generate_sec") or 0.0),
                    exec=float(trace.get("total_exec_sec") or 0.0),
                )
            )
    if count:
        print(f"\nfinal_count={count} avg_reward={sum(rewards) / len(rewards):.4f}")
    else:
        print("No final events found.", file=sys.stderr)
    return 0 if count else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path, help="Path to terminal_transcripts.jsonl")
    parser.add_argument("--follow", "-f", action="store_true", help="Follow the file like tail -f")
    parser.add_argument("--finals", action="store_true", help="Only print final trajectory summaries")
    parser.add_argument("--events", nargs="*", help="Only print these event types, e.g. assistant feedback final")
    parser.add_argument("--max-chars", type=int, default=0, help="Per-content display cap; 0 means no cap")
    parser.add_argument(
        "--full-final-messages",
        action="store_true",
        help="Print the full message list for final events instead of only the summary",
    )
    args = parser.parse_args()

    if args.finals:
        return _print_final_summary(args.jsonl)

    event_filter = set(args.events or [])
    for line in _iter_lines(args.jsonl, args.follow):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[bad-json] {exc}: {line.rstrip()}", file=sys.stderr)
            continue
        if event_filter and event.get("event") not in event_filter:
            continue
        _print_event(event, args)
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
