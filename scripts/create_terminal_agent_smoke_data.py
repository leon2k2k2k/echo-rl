#!/usr/bin/env python3
"""Create tiny terminal-agent parquet files for ECHO smoke tests."""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SYSTEM_PROMPT = (
    "You are a highly capable Linux terminal agent. Complete the user's task by "
    "running commands and verifying the result. When the task is complete, call done."
)

TOOL_INSTRUCTIONS = """You are given a task description and your goal is to solve the task by using shell commands and python code.
Start each response with a <think>...</think> section where you analyze the current state based on the terminal output and describe your plan for the next steps. Then, provide your commands using bash tool calls that specify the commands to execute. When you determine that the task is complete, use the done tool call to indicate completion.

Required tools:
- "bash": Call this to execute the specified bash command in the terminal and return the output to you in the next turn as terminal_output. Example:
<tool_call>
<function=bash>
<parameter=command>
echo hello
</parameter>
</function>
</tool_call>

Optional tools:
- "done": Call this when you have verified the task is solved.
<tool_call>
<function=done>
</function>
</tool_call>

IMPORTANT:
- Only use <think> or <tool_call> to structure your response as described above.
- Exactly one <think> ... </think> block should be present.
- The bash tool call command string is used verbatim.
"""


def _add_text(tf: tarfile.TarFile, name: str, text: str, mode: int = 0o644) -> None:
    data = text.encode("utf-8")
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    tf.addfile(info, io.BytesIO(data))


def _task_archive(task_id: int) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        _add_text(
            tf,
            "task.toml",
            f"""
[task]
name = "echo/smoke-{task_id}"
authors = [{{ name = "ECHO smoke", email = "echo-smoke@example.com" }}]

[environment]
build_timeout_sec = 120.0

[agent]
timeout_sec = 60.0

[verifier]
timeout_sec = 30.0
""".lstrip(),
        )
        _add_text(
            tf,
            "instruction.md",
            "Create /app/answer.txt containing exactly: smoke-ok\n",
        )
        _add_text(
            tf,
            "environment/Dockerfile",
            "FROM ubuntu:24.04\nWORKDIR /app\n",
        )
        _add_text(
            tf,
            "tests/test.sh",
            """#!/bin/bash
set -euo pipefail
if [ "$(cat /app/answer.txt 2>/dev/null)" = "smoke-ok" ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
""",
            mode=0o755,
        )
    return buf.getvalue()


def _prompt(task_id: int) -> list[dict[str, str]]:
    instruction = (
        f"{TOOL_INSTRUCTIONS}\n\nTask: Create /app/answer.txt containing exactly "
        f"smoke-ok. This is smoke task {task_id}."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]


def _write_dataset(path: Path, rows: int, *, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prompts = []
    paths = []
    task_binaries = []
    for idx in range(rows):
        task_id = offset + idx
        prompts.append(_prompt(task_id))
        paths.append(f"echo-smoke-{task_id}")
        task_binaries.append(_task_archive(task_id))

    table = pa.table(
        {
            "prompt": pa.array(prompts),
            "path": pa.array(paths, type=pa.string()),
            "task_binary": pa.array(task_binaries, type=pa.binary()),
        }
    )
    pq.write_table(table, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--val", required=True, type=Path)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()

    _write_dataset(args.train, args.rows, offset=0)
    _write_dataset(args.val, args.rows, offset=10_000)
    print(f"Wrote train smoke parquet: {args.train}")
    print(f"Wrote val smoke parquet: {args.val}")


if __name__ == "__main__":
    main()
