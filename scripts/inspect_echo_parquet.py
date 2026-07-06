#!/usr/bin/env python3
"""Inspect an ECHO terminal-agent parquet file."""

from __future__ import annotations

import argparse
import io
import tarfile

import pyarrow.parquet as pq


def _preview(text: str, limit: int) -> str:
    text = text.replace("\n", "\\n")
    return text[:limit] + ("..." if len(text) > limit else "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--prompt-chars", type=int, default=700)
    parser.add_argument("--tar-files", type=int, default=30)
    args = parser.parse_args()

    table = pq.read_table(args.parquet)
    print(f"path={args.parquet}")
    print(f"rows={table.num_rows}")
    print("schema:")
    print(table.schema)

    for i, row in enumerate(table.slice(0, min(args.rows, table.num_rows)).to_pylist()):
        print(f"\n--- row {i} ---")
        print(f"path: {row.get('path')}")
        prompt = row.get("prompt") or []
        print(f"prompt_messages: {len(prompt)}")
        for j, message in enumerate(prompt[:2]):
            print(f"prompt[{j}].role: {message.get('role')}")
            print(f"prompt[{j}].content: {_preview(str(message.get('content', '')), args.prompt_chars)}")

        task_binary = row.get("task_binary")
        print(f"task_binary_type: {type(task_binary).__name__}")
        print(f"task_binary_bytes: {len(task_binary) if task_binary is not None else 0}")
        if task_binary:
            with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as tf:
                names = [member.name for member in tf.getmembers()]
            print("task_binary_files:")
            for name in names[: args.tar_files]:
                print(f"  {name}")
            if len(names) > args.tar_files:
                print(f"  ... ({len(names) - args.tar_files} more)")


if __name__ == "__main__":
    main()
