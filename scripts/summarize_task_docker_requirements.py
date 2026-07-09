#!/usr/bin/env python3
"""Summarize Dockerfile requirements embedded in ECHO terminal-agent parquets."""

from __future__ import annotations

import argparse
import collections
import io
import re
import tarfile
from dataclasses import dataclass

import pyarrow.parquet as pq


FROM_RE = re.compile(r"^\s*FROM\s+([^\s]+)", re.IGNORECASE)
APT_RE = re.compile(r"\b(?:apt-get|apt)\s+install\b([^;&|]*)", re.IGNORECASE)
PIP_RE = re.compile(r"\b(?:pip3?|python3?\s+-m\s+pip)\s+install\b([^;&|]*)", re.IGNORECASE)


@dataclass
class DockerSummary:
    idx: int
    path: str
    source: str
    difficulty: str
    base_images: list[str]
    apt_packages: list[str]
    pip_packages: list[str]
    has_curl_or_wget: bool
    has_git_clone: bool
    has_network_install: bool
    dockerfile: str


def _read_member(task_binary: bytes, name: str) -> str:
    with tarfile.open(fileobj=io.BytesIO(task_binary), mode="r:*") as tf:
        member = tf.extractfile(name)
        if member is None:
            return ""
        return member.read().decode("utf-8", errors="replace")


def _clean_packages(raw: str) -> list[str]:
    raw = raw.replace("\\\n", " ")
    tokens = []
    for token in raw.split():
        token = token.strip("\\")
        if not token or token.startswith("-"):
            continue
        if token in {"&&", "||", ";"}:
            break
        if "=" in token:
            token = token.split("=", 1)[0]
        tokens.append(token)
    return tokens


def _summarize_row(idx: int, row: dict) -> DockerSummary:
    dockerfile = _read_member(bytes(row["task_binary"]), "environment/Dockerfile")
    base_images = []
    apt_packages = []
    pip_packages = []
    for line in dockerfile.splitlines():
        from_match = FROM_RE.search(line)
        if from_match:
            base_images.append(from_match.group(1))
        for match in APT_RE.finditer(line):
            apt_packages.extend(_clean_packages(match.group(1)))
        for match in PIP_RE.finditer(line):
            pip_packages.extend(_clean_packages(match.group(1)))

    lowered = dockerfile.lower()
    has_curl_or_wget = bool(re.search(r"\b(curl|wget)\b", lowered))
    has_git_clone = "git clone" in lowered
    has_network_install = bool(apt_packages or pip_packages or has_curl_or_wget or has_git_clone)
    return DockerSummary(
        idx=idx,
        path=str(row.get("path", "")),
        source=str(row.get("source", "")),
        difficulty=str(row.get("difficulty", "")),
        base_images=base_images,
        apt_packages=apt_packages,
        pip_packages=pip_packages,
        has_curl_or_wget=has_curl_or_wget,
        has_git_clone=has_git_clone,
        has_network_install=has_network_install,
        dockerfile=dockerfile,
    )


def _counter(items: list[list[str]]) -> collections.Counter[str]:
    counter: collections.Counter[str] = collections.Counter()
    for values in items:
        counter.update(values)
    return counter


def _print_counter(title: str, counter: collections.Counter[str], limit: int) -> None:
    print(f"\n{title}:")
    if not counter:
        print("  <none>")
        return
    for key, count in counter.most_common(limit):
        print(f"  {count:4d}  {key}")


def _preview_dockerfile(text: str, max_lines: int) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[:max_lines])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet")
    parser.add_argument("--rows", type=int, default=0, help="Rows to inspect; 0 means all rows.")
    parser.add_argument("--examples", type=int, default=8)
    parser.add_argument("--dockerfile-lines", type=int, default=40)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    table = pq.read_table(args.parquet)
    if args.rows > 0:
        table = table.slice(0, min(args.rows, table.num_rows))
    rows = table.to_pylist()
    summaries = [_summarize_row(i, row) for i, row in enumerate(rows)]

    print(f"parquet={args.parquet}")
    print(f"rows_inspected={len(summaries)}")
    print(f"rows_with_network_install={sum(s.has_network_install for s in summaries)}")
    print(f"rows_with_curl_or_wget={sum(s.has_curl_or_wget for s in summaries)}")
    print(f"rows_with_git_clone={sum(s.has_git_clone for s in summaries)}")

    source_counts = collections.Counter((s.source or "<missing>", s.difficulty or "<missing>") for s in summaries)
    print("\nsource/difficulty:")
    for (source, difficulty), count in sorted(source_counts.items()):
        print(f"  {count:4d}  {source}/{difficulty}")

    _print_counter("base images", _counter([s.base_images for s in summaries]), args.top)
    _print_counter("apt packages", _counter([s.apt_packages for s in summaries]), args.top)
    _print_counter("pip packages", _counter([s.pip_packages for s in summaries]), args.top)

    risky = [s for s in summaries if s.has_network_install]
    print(f"\nexamples with network/build requirements: {len(risky)}")
    for summary in risky[: args.examples]:
        print("\n---")
        print(f"row={summary.idx} path={summary.path} source={summary.source} difficulty={summary.difficulty}")
        print(f"base_images={summary.base_images or []}")
        print(f"apt_packages={summary.apt_packages or []}")
        print(f"pip_packages={summary.pip_packages or []}")
        print(f"curl_or_wget={summary.has_curl_or_wget} git_clone={summary.has_git_clone}")
        print("Dockerfile:")
        print(_preview_dockerfile(summary.dockerfile, args.dockerfile_lines))


if __name__ == "__main__":
    main()
