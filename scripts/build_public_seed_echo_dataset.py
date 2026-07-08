#!/usr/bin/env python3
"""Build ECHO terminal-agent parquets from public seed task datasets.

Supported sources:
- Endless Terminals snapshot directory from obiwan96/endless-terminals.
- OpenThoughts-Agent-v1-RL tasks.parquet.

The output rows match echo_rl.terminal_agent.dataset.TerminalAgentTaskDataset:
prompt, path, task_binary. Extra source/difficulty/category columns are included
for inspection and stratified splits.
"""

from __future__ import annotations

import argparse
import io
import random
import tarfile
import sys
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from echo_rl.terminal_agent.prompts import QWEN35_INSTRUCTION_PREFIX


SYSTEM_PROMPT = (
    "You are a highly capable Linux terminal agent. Complete the user's task by "
    "running commands and verifying the result. When the task is complete, call done."
)

REQUIRED_TASK_FILES = {
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "tests/test.sh",
}


def _prompt_for_instruction(instruction: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{QWEN35_INSTRUCTION_PREFIX}\n\nTask: {instruction.strip()}"},
    ]


def _safe_member_name(path: Path, root: Path) -> str:
    rel = str(PurePosixPath(path.relative_to(root).as_posix()))
    if not rel or rel.startswith("../") or "/../" in rel:
        raise ValueError(f"Unsafe archive member path: {path}")
    return rel


def _pack_task_dir(task_dir: Path) -> bytes:
    files = [p for p in task_dir.rglob("*") if p.is_file()]
    names = {_safe_member_name(p, task_dir) for p in files}
    missing = REQUIRED_TASK_FILES - names
    if missing:
        raise ValueError(f"{task_dir} is missing required files: {sorted(missing)}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in sorted(files):
            name = _safe_member_name(path, task_dir)
            tf.add(path, arcname=name, recursive=False)
    return buf.getvalue()


def _read_toml_metadata(task_dir: Path) -> dict[str, str]:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py3.10 fallback if tomli is installed
        import tomli as tomllib  # type: ignore

    data = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    metadata = data.get("metadata") or {}
    return {
        "difficulty": str(metadata.get("difficulty") or ""),
        "category": str(metadata.get("category") or ""),
    }


def _rows_from_endless(endless_dir: Path, *, limit: int = 0) -> list[dict]:
    task_dirs = sorted(p for p in endless_dir.iterdir() if p.is_dir() and (p / "task.toml").exists())
    if limit > 0:
        task_dirs = task_dirs[:limit]
    rows = []
    for task_dir in task_dirs:
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8").strip()
        metadata = _read_toml_metadata(task_dir)
        rows.append(
            {
                "prompt": _prompt_for_instruction(instruction),
                "path": f"endless/{task_dir.name}",
                "task_binary": _pack_task_dir(task_dir),
                "source": "endless",
                "difficulty": metadata["difficulty"] or "easy",
                "category": metadata["category"] or "programming",
            }
        )
    return rows


def _task_binary_names(task_binary: bytes) -> set[str]:
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as tf:
        return {member.name for member in tf.getmembers() if member.isfile()}


def _read_from_task_binary(task_binary: bytes, name: str) -> str:
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as tf:
        fileobj = tf.extractfile(name)
        if fileobj is None:
            raise ValueError(f"Missing {name} in task_binary")
        return fileobj.read().decode("utf-8").strip()


def _rows_from_openthoughts(openthoughts_parquet: Path, *, limit: int = 0) -> list[dict]:
    table = pq.read_table(openthoughts_parquet)
    raw_rows = table.to_pylist()
    if limit > 0:
        raw_rows = raw_rows[:limit]
    rows = []
    for row in raw_rows:
        task_binary = bytes(row["task_binary"])
        names = _task_binary_names(task_binary)
        missing = REQUIRED_TASK_FILES - names
        if missing:
            raise ValueError(f"{row['path']} is missing required files: {sorted(missing)}")
        instruction = _read_from_task_binary(task_binary, "instruction.md")
        rows.append(
            {
                "prompt": _prompt_for_instruction(instruction),
                "path": f"openthoughts/{row['path']}",
                "task_binary": task_binary,
                "source": "openthoughts",
                "difficulty": "medium",
                "category": "terminal-automation",
            }
        )
    return rows


def _stratified_val_indices(rows: list[dict], val_size: int, seed: int) -> set[int]:
    if val_size <= 0:
        return set()
    if val_size >= len(rows):
        raise ValueError(f"val_size={val_size} must be smaller than row count {len(rows)}")

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[int]] = {}
    for idx, row in enumerate(rows):
        buckets.setdefault((row["source"], row["difficulty"]), []).append(idx)

    selected: set[int] = set()
    remaining = val_size
    bucket_items = sorted(buckets.items(), key=lambda item: item[0])
    for bucket_idx, (_, indices) in enumerate(bucket_items):
        rng.shuffle(indices)
        if bucket_idx == len(bucket_items) - 1:
            take = remaining
        else:
            take = round(val_size * len(indices) / len(rows))
            take = min(take, remaining)
        selected.update(indices[:take])
        remaining -= take

    if len(selected) < val_size:
        pool = [idx for idx in range(len(rows)) if idx not in selected]
        rng.shuffle(pool)
        selected.update(pool[: val_size - len(selected)])
    elif len(selected) > val_size:
        selected = set(sorted(selected)[:val_size])
    return selected


def _write_parquet(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "prompt": pa.array([row["prompt"] for row in rows]),
            "path": pa.array([row["path"] for row in rows], type=pa.string()),
            "task_binary": pa.array([row["task_binary"] for row in rows], type=pa.binary()),
            "source": pa.array([row["source"] for row in rows], type=pa.string()),
            "difficulty": pa.array([row["difficulty"] for row in rows], type=pa.string()),
            "category": pa.array([row["category"] for row in rows], type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _counts(rows: list[dict]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["source"], row["difficulty"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endless-dir", type=Path, help="Local snapshot of obiwan96/endless-terminals.")
    parser.add_argument("--openthoughts-parquet", type=Path, help="OpenThoughts-Agent-v1-RL tasks.parquet.")
    parser.add_argument("--train-out", required=True, type=Path)
    parser.add_argument("--val-out", required=True, type=Path)
    parser.add_argument("--val-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-endless", type=int, default=0)
    parser.add_argument("--max-openthoughts", type=int, default=0)
    args = parser.parse_args()

    rows: list[dict] = []
    if args.endless_dir:
        rows.extend(_rows_from_endless(args.endless_dir, limit=args.max_endless))
    if args.openthoughts_parquet:
        rows.extend(_rows_from_openthoughts(args.openthoughts_parquet, limit=args.max_openthoughts))
    if not rows:
        raise ValueError("At least one of --endless-dir or --openthoughts-parquet is required.")

    random.Random(args.seed).shuffle(rows)
    val_indices = _stratified_val_indices(rows, args.val_size, args.seed)
    train_rows = [row for idx, row in enumerate(rows) if idx not in val_indices]
    val_rows = [row for idx, row in enumerate(rows) if idx in val_indices]

    _write_parquet(args.train_out, train_rows)
    _write_parquet(args.val_out, val_rows)

    print(f"total rows: {len(rows)}")
    print("counts by source/difficulty:")
    for (source, difficulty), count in sorted(_counts(rows).items()):
        print(f"  {source}/{difficulty}: {count}")
    print(f"train rows: {len(train_rows)}")
    print(f"val rows: {len(val_rows)}")
    print(f"wrote train: {args.train_out}")
    print(f"wrote val: {args.val_out}")


if __name__ == "__main__":
    main()
