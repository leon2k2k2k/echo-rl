#!/usr/bin/env bash
set -euo pipefail

# Filter the public combined seed dataset to terminal tasks based on ubuntu:22.04.
# This excludes OpenThoughts rows that currently require ubuntu:24.04.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export SOURCE_ROOT="${SOURCE_ROOT:-$ECHO_RUNTIME_ROOT/data/public_seed_echo}"
export OUT_ROOT="${OUT_ROOT:-$ECHO_RUNTIME_ROOT/data/ubuntu22_seed_echo}"
export BASE_IMAGE="${BASE_IMAGE:-ubuntu:22.04}"

mkdir -p "$OUT_ROOT"

"$PYTHON" scripts/filter_terminal_agent_parquet_by_base_image.py \
  "$SOURCE_ROOT/train.parquet" \
  "$OUT_ROOT/train.parquet" \
  --base-image "$BASE_IMAGE"

"$PYTHON" scripts/filter_terminal_agent_parquet_by_base_image.py \
  "$SOURCE_ROOT/val.parquet" \
  "$OUT_ROOT/val.parquet" \
  --base-image "$BASE_IMAGE"

echo
echo "Ubuntu-22-only dataset counts:"
"$PYTHON" -c 'import collections, os, pyarrow.parquet as pq; p=os.environ["OUT_ROOT"]+"/train.parquet"; rows=pq.read_table(p).to_pylist(); print("train", len(rows), collections.Counter((r.get("source"), r.get("difficulty")) for r in rows))'
"$PYTHON" -c 'import collections, os, pyarrow.parquet as pq; p=os.environ["OUT_ROOT"]+"/val.parquet"; rows=pq.read_table(p).to_pylist(); print("val", len(rows), collections.Counter((r.get("source"), r.get("difficulty")) for r in rows))'

echo
echo "Docker requirement summary:"
"$PYTHON" scripts/summarize_task_docker_requirements.py "$OUT_ROOT/val.parquet" --rows 100 --examples 3 || true

cat <<EOF

Ubuntu-22-only dataset ready:
  train: $OUT_ROOT/train.parquet
  val:   $OUT_ROOT/val.parquet
EOF
