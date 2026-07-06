#!/usr/bin/env python3
"""Convert TMax Harbor tasks into ECHO terminal-agent parquet files."""

from __future__ import annotations

import argparse
import io
import random
import tarfile
import zipfile
from pathlib import PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

from echo_rl.terminal_agent.prompts import QWEN35_INSTRUCTION_PREFIX


REQUIRED_TASK_FILES = {
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
}

SYSTEM_PROMPT = (
    "You are a highly capable Linux terminal agent. Complete the user's task by "
    "running commands and verifying the result. When the task is complete, call done."
)


def _row_get(row: dict, *keys: str):
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return None


def _load_rows(metadata_parquet: str) -> list[dict]:
    table = pq.read_table(metadata_parquet)
    return table.to_pylist()


def _task_name_from_row(row: dict) -> str:
    task_id = _row_get(row, "task_id", "id", "name", "path")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"Could not find task id in row with keys: {sorted(row)}")
    return task_id


def _zip_dir_for_task(task_id: str) -> str:
    return task_id.rstrip("/").split("/")[-1]


def _instruction_from_row(row: dict) -> str | None:
    value = _row_get(row, "instruction", "prompt", "task", "question")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _read_instruction_from_zip(zf: zipfile.ZipFile, task_dir: str) -> str:
    path = f"{task_dir}/instruction.md"
    try:
        with zf.open(path) as f:
            return f.read().decode("utf-8").strip()
    except KeyError as exc:
        raise ValueError(f"Missing {path} in tasks zip") from exc


def _task_binary_from_zip(zf: zipfile.ZipFile, task_dir: str) -> bytes:
    prefix = f"{task_dir.rstrip('/')}/"
    names = [name for name in zf.namelist() if name.startswith(prefix) and not name.endswith("/")]
    if not names:
        raise ValueError(f"No files found for task directory {task_dir!r} in tasks zip")

    stripped_names = {str(PurePosixPath(name[len(prefix) :])) for name in names}
    missing = REQUIRED_TASK_FILES - stripped_names
    if missing:
        raise ValueError(f"Task {task_dir!r} is missing required files: {sorted(missing)}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for zip_name in names:
            rel_name = str(PurePosixPath(zip_name[len(prefix) :]))
            if not rel_name or rel_name.startswith("../"):
                continue
            info = zf.getinfo(zip_name)
            tar_info = tarfile.TarInfo(rel_name)
            tar_info.size = info.file_size
            tar_info.mode = (info.external_attr >> 16) & 0o777 or 0o644
            with zf.open(zip_name) as src:
                tf.addfile(tar_info, src)
    return buf.getvalue()


def _prompt_for_instruction(instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{QWEN35_INSTRUCTION_PREFIX}\n\nTask: {instruction}"},
    ]


def _build_echo_rows(rows: list[dict], tasks_zip: str, indices: list[int]) -> list[dict]:
    output = []
    with zipfile.ZipFile(tasks_zip) as zf:
        for idx in indices:
            row = rows[idx]
            task_id = _task_name_from_row(row)
            task_dir = _zip_dir_for_task(task_id)
            instruction = _instruction_from_row(row) or _read_instruction_from_zip(zf, task_dir)
            output.append(
                {
                    "prompt": _prompt_for_instruction(instruction),
                    "path": task_id,
                    "task_binary": _task_binary_from_zip(zf, task_dir),
                }
            )
    return output


def _write_parquet(path: str, rows: list[dict]) -> None:
    table = pa.table(
        {
            "prompt": pa.array([row["prompt"] for row in rows]),
            "path": pa.array([row["path"] for row in rows], type=pa.string()),
            "task_binary": pa.array([row["task_binary"] for row in rows], type=pa.binary()),
        }
    )
    pq.write_table(table, path)


def _validate_output(path: str, n: int) -> None:
    rows = pq.read_table(path).to_pylist()
    for row in rows[:n]:
        with tarfile.open(fileobj=io.BytesIO(row["task_binary"]), mode="r:gz") as tf:
            names = {member.name for member in tf.getmembers() if member.isfile()}
        missing = REQUIRED_TASK_FILES - names
        if missing:
            raise ValueError(f"Converted row {row['path']!r} is missing files: {sorted(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-parquet", required=True)
    parser.add_argument("--tasks-zip", required=True)
    parser.add_argument("--train-out", required=True)
    parser.add_argument("--val-out", required=True)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional cap before splitting; 0 means all rows.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validate-rows", type=int, default=5)
    args = parser.parse_args()

    rows = _load_rows(args.metadata_parquet)
    if args.max_rows > 0:
        rows = rows[: args.max_rows]
    if len(rows) <= args.val_size:
        raise ValueError(f"Need more rows than val-size={args.val_size}, got {len(rows)}")

    indices = list(range(len(rows)))
    random.Random(args.seed).shuffle(indices)
    val_indices = indices[: args.val_size]
    train_indices = indices[args.val_size :]

    train_rows = _build_echo_rows(rows, args.tasks_zip, train_indices)
    val_rows = _build_echo_rows(rows, args.tasks_zip, val_indices)

    _write_parquet(args.train_out, train_rows)
    _write_parquet(args.val_out, val_rows)
    _validate_output(args.train_out, args.validate_rows)
    _validate_output(args.val_out, args.validate_rows)

    print(f"Wrote train parquet: {args.train_out} ({len(train_rows)} rows)")
    print(f"Wrote val parquet: {args.val_out} ({len(val_rows)} rows)")


if __name__ == "__main__":
    main()
