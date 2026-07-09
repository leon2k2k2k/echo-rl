#!/usr/bin/env python3
"""Filter ECHO terminal-agent parquet rows by Docker base image."""

from __future__ import annotations

import argparse
import io
import re
import tarfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


FROM_RE = re.compile(r"^\s*FROM\s+([^\s#]+)", re.IGNORECASE | re.MULTILINE)


def _read_member(task_binary: bytes, name: str) -> str:
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as tf:
        member = tf.extractfile(name)
        if member is None:
            return ""
        return member.read().decode("utf-8", errors="replace")


def _base_images(row: dict) -> list[str]:
    dockerfile = _read_member(bytes(row["task_binary"]), "environment/Dockerfile")
    return FROM_RE.findall(dockerfile)


def filter_parquet(input_path: Path, output_path: Path, base_image: str) -> tuple[int, int]:
    table = pq.read_table(input_path)
    rows = table.to_pylist()
    kept_indices = [
        idx for idx, row in enumerate(rows)
        if base_image in _base_images(row)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = table.take(pa.array(kept_indices, type=pa.int64()))
    pq.write_table(filtered, output_path)
    return len(rows), len(kept_indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base-image", default="ubuntu:22.04")
    args = parser.parse_args()

    total, kept = filter_parquet(args.input, args.output, args.base_image)
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"base_image={args.base_image}")
    print(f"rows_in={total}")
    print(f"rows_kept={kept}")


if __name__ == "__main__":
    main()
