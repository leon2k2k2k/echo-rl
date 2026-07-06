#!/usr/bin/env python3
"""Summarize an ECHO smoke failure from Slurm and collected Ray artifacts."""

from __future__ import annotations

import argparse
import gzip
import io
import re
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEATH_RE = re.compile(
    r"Worker ID: (?P<worker_id>[0-9a-f]+).*?Worker PID: (?P<pid>\d+).*?Worker exit type: (?P<exit_type>\S+)",
    re.DOTALL,
)
ACTOR_RE = re.compile(r"actor_id: (?P<actor_id>[0-9a-f]+).*?\npid: (?P<pid>\d+)", re.DOTALL)
STAGE_RE = re.compile(
    r"(Started:|Finished:|reward/|batch_num_seq|avg_final_rewards|Traceback|ActorDiedError|RayTaskError|OutOfMemory|SYSTEM_ERROR|SIGKILL|policy_train|train_critic_and_policy|Finished: 'step')"
)
ERROR_RE = re.compile(
    r"(Traceback|RuntimeError|ActorDiedError|RayTaskError|OutOfMemory|CUDA out of memory|SYSTEM_ERROR|SIGKILL|SIGSEGV|NCCL|illegal memory|segmentation fault|Killed)",
    re.IGNORECASE,
)
POLICY_RE = re.compile(
    r"(\[policy-train\]|forward_backward|optim_step|aux_loss|final_loss|policy_loss|world_loss|nextlat_loss)"
)


@dataclass(frozen=True)
class DeadWorker:
    worker_id: str | None
    pid: str
    exit_type: str | None = None
    actor_id: str | None = None


def iter_text_files(paths: Iterable[Path]) -> Iterable[tuple[str, str]]:
    for path in paths:
        if not path.exists() or not path.is_file():
            continue
        try:
            yield str(path), path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            yield str(path), f"[could not read: {exc}]"


def tail_lines(text: str, n: int) -> str:
    return "\n".join(text.splitlines()[-n:])


def print_section(title: str) -> None:
    print(f"\n===== {title} =====")


def find_slurm_logs(output_dir: Path, log_dir: Path | None, job_id: str | None, include_recent: bool) -> list[Path]:
    candidates: list[Path] = []
    matched_job_logs = False
    if log_dir is not None:
        if job_id:
            job_logs = sorted(log_dir.glob(f"*{job_id}*.out")) + sorted(log_dir.glob(f"*{job_id}*.err"))
            matched_job_logs = bool(job_logs)
            candidates.extend(job_logs)
        if include_recent or not matched_job_logs:
            candidates.extend(sorted(log_dir.glob("echo-rl-smoke-*.out"))[-3:])
            candidates.extend(sorted(log_dir.glob("echo-rl-smoke-*.err"))[-3:])
    candidates.extend(output_dir.glob("*.log"))
    seen = set()
    unique = []
    for path in candidates:
        resolved = str(path)
        if resolved not in seen and path.exists():
            unique.append(path)
            seen.add(resolved)
    return unique


def extract_dead_workers(texts: Iterable[tuple[str, str]]) -> list[DeadWorker]:
    by_pid: dict[str, DeadWorker] = {}
    for _, text in texts:
        for match in DEATH_RE.finditer(text):
            worker = DeadWorker(
                worker_id=match.group("worker_id"),
                pid=match.group("pid"),
                exit_type=match.group("exit_type"),
            )
            by_pid[worker.pid] = worker
        for match in ACTOR_RE.finditer(text):
            pid = match.group("pid")
            previous = by_pid.get(pid)
            by_pid[pid] = DeadWorker(
                worker_id=previous.worker_id if previous else None,
                pid=pid,
                exit_type=previous.exit_type if previous else None,
                actor_id=match.group("actor_id"),
            )
    return sorted(by_pid.values(), key=lambda worker: int(worker.pid))


def print_stage_timeline(texts: Iterable[tuple[str, str]], max_lines: int) -> None:
    hits: list[str] = []
    for name, text in texts:
        for line in text.splitlines():
            if STAGE_RE.search(line):
                hits.append(f"{Path(name).name}: {line}")
    if not hits:
        print("No stage lines found.")
        return
    print("\n".join(hits[-max_lines:]))


def tar_members_by_name(tar_path: Path) -> dict[str, tarfile.TarInfo]:
    if not tar_path.exists():
        return {}
    with tarfile.open(tar_path, "r:gz") as tar:
        return {member.name: member for member in tar.getmembers() if member.isfile()}


def read_tar_member(tar_path: Path, member_name: str) -> str:
    with tarfile.open(tar_path, "r:gz") as tar:
        member = tar.getmember(member_name)
        f = tar.extractfile(member)
        if f is None:
            return ""
        data = f.read()
    return data.decode("utf-8", errors="replace")


def matching_ray_members(tar_path: Path, workers: list[DeadWorker]) -> list[str]:
    members = tar_members_by_name(tar_path)
    if not members:
        return []
    needles = set()
    for worker in workers:
        needles.add(worker.pid)
        if worker.worker_id:
            needles.add(worker.worker_id)
    matches = []
    for name in members:
        if any(needle and needle in name for needle in needles):
            matches.append(name)
    return sorted(matches)


def print_dead_worker_logs(tar_path: Path, workers: list[DeadWorker], tail: int) -> None:
    if not tar_path.exists():
        print(f"No Ray tarball found at {tar_path}")
        return
    matches = matching_ray_members(tar_path, workers)
    if not matches:
        print("No Ray log files matched the dead worker IDs/PIDs.")
        return
    for name in matches:
        text = read_tar_member(tar_path, name)
        print_section(name)
        if text.strip():
            print(tail_lines(text, tail))
        else:
            print("[empty log]")


def print_error_search(tar_path: Path, tail: int) -> None:
    if not tar_path.exists():
        return
    hits: list[str] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.size == 0:
                continue
            if not member.name.endswith((".err", ".out", ".log", ".txt")):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            text = f.read().decode("utf-8", errors="replace")
            for line in text.splitlines():
                if ERROR_RE.search(line):
                    hits.append(f"{member.name}: {line}")
    if hits:
        print("\n".join(hits[-tail:]))
    else:
        print("No explicit OOM/SIG/NCCL/Python traceback lines found in collected Ray logs.")


def print_policy_train_search(tar_path: Path, tail: int) -> None:
    if not tar_path.exists():
        return
    hits: list[str] = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or member.size == 0:
                continue
            if not member.name.endswith((".err", ".out", ".log", ".txt")):
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            text = f.read().decode("utf-8", errors="replace")
            for line in text.splitlines():
                if POLICY_RE.search(line):
                    hits.append(f"{member.name}: {line}")
    if hits:
        print("\n".join(hits[-tail:]))
    else:
        print("No policy-train worker lines found in collected Ray logs.")


def maybe_print_gzip_tail(path: Path, title: str, tail: int) -> None:
    if not path.exists():
        return
    print_section(title)
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            print(tail_lines(f.read(), tail))
    except OSError:
        print(path.read_text(encoding="utf-8", errors="replace")[-4000:])


def infer_job_id(output_dir: Path, explicit: str | None) -> str | None:
    if explicit:
        return explicit
    match = re.search(r"(\d{5,})", output_dir.name)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Run output dir, e.g. .../outputs/echo-base-smoke-373938")
    parser.add_argument("--job-id", help="Slurm job id. Inferred from output dir when possible.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("/home/fit/alex/WORK/leon/echo_rl_runtime/logs"),
        help="Directory containing echo-rl-smoke-*.out/.err",
    )
    parser.add_argument("--tail", type=int, default=120, help="Lines to show from each relevant log")
    parser.add_argument(
        "--include-recent",
        action="store_true",
        help="Also include the latest echo-rl-smoke logs even when --job-id/output dir identifies one job.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    debug_dir = output_dir / "debug"
    job_id = infer_job_id(output_dir, args.job_id)
    slurm_logs = find_slurm_logs(output_dir, args.log_dir, job_id, args.include_recent)
    slurm_texts = list(iter_text_files(slurm_logs))

    print_section("run")
    print(f"output_dir={output_dir}")
    print(f"job_id={job_id}")
    print("slurm_logs=" + (", ".join(str(path) for path in slurm_logs) if slurm_logs else "none"))

    print_section("stage timeline")
    print_stage_timeline(slurm_texts, args.tail)

    workers = extract_dead_workers(slurm_texts)
    print_section("dead workers")
    if workers:
        for worker in workers:
            print(
                f"pid={worker.pid} worker_id={worker.worker_id or '-'} "
                f"actor_id={worker.actor_id or '-'} exit_type={worker.exit_type or '-'}"
            )
    else:
        print("No dead workers found in Slurm logs.")

    ray_index = debug_dir / "ray_log_index.txt"
    if ray_index.exists():
        print_section("ray log index matches")
        index_text = ray_index.read_text(encoding="utf-8", errors="replace")
        needles = [worker.pid for worker in workers] + [worker.worker_id for worker in workers if worker.worker_id]
        lines = [
            line
            for line in index_text.splitlines()
            if not needles or any(needle and needle in line for needle in needles)
        ]
        print("\n".join(lines[-args.tail:]) if lines else "No index lines matched dead workers.")

    ray_tar = debug_dir / "ray_logs.tgz"
    print_section("dead worker ray logs")
    print_dead_worker_logs(ray_tar, workers, args.tail)

    print_section("ray error search")
    print_error_search(ray_tar, args.tail)

    print_section("ray policy-train search")
    print_policy_train_search(ray_tar, args.tail)

    final_snapshot = debug_dir / "final_snapshot.log"
    if final_snapshot.exists():
        print_section("final snapshot tail")
        print(tail_lines(final_snapshot.read_text(encoding="utf-8", errors="replace"), args.tail))

    heartbeat = debug_dir / "heartbeat.log"
    if heartbeat.exists():
        print_section("heartbeat tail")
        print(tail_lines(heartbeat.read_text(encoding="utf-8", errors="replace"), args.tail))

    print_section("interpretation")
    print(
        "If the dead worker logs are empty and Ray only reports SYSTEM_ERROR/EOF, the process likely died below "
        "Python: SIGKILL from host OOM/cgroup pressure, CUDA/NCCL/native crash, or external termination. "
        "If Python/CUDA tracebacks appear above, use those as the primary failure cause."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
