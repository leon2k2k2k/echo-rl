#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/transfer_pass16_debug_bundle.sh JOB_ID [JOB_ID ...]

Dumps a compact pass@16 debug bundle, commits it to the artifact branch, and
pushes it. This uses a temporary git worktree, so the main training checkout
does not need to be clean and does not switch branches.

Environment:
  REMOTE=private                         Git remote to fetch/push.
  BRANCH=data-terminal-artifacts         Artifact branch to update.
  ARTIFACT_ROOT=pass16_debug_bundles     Directory on artifact branch.
  KEEP_ARTIFACT_WORKTREE=1               Keep temporary worktree for debugging.

Example:
  bash scripts/transfer_pass16_debug_bundle.sh 375559 375560
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -eq 0 ]]; then
  usage
  exit $([[ "$#" -eq 0 ]] && echo 2 || echo 0)
fi

for job_id in "$@"; do
  if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: JOB_ID must be numeric, got: $job_id" >&2
    exit 2
  fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

remote="${REMOTE:-private}"
branch="${BRANCH:-data-terminal-artifacts}"
artifact_root="${ARTIFACT_ROOT:-pass16_debug_bundles}"
stamp="$(date +%Y%m%d-%H%M%S)"
job_label="$(printf '%s-' "$@" | sed 's/-$//')"
worktree="$ECHO_RUNTIME_ROOT/artifact_worktrees/pass16-debug-${job_label}-${stamp}"

echo "REMOTE=$remote"
echo "BRANCH=$branch"
echo "JOBS=$*"

dump_output="$(
  bash scripts/dump_pass16_debug_bundle.sh "$@"
)"
echo "$dump_output"

bundle_dir="$(
  awk -F= '/^bundle_dir=/{print $2}' <<<"$dump_output" | tail -1
)"

if [[ -z "$bundle_dir" || ! -d "$bundle_dir" ]]; then
  echo "ERROR: could not locate bundle_dir from dump output" >&2
  exit 1
fi

bundle_name="$(basename "$bundle_dir")"

cleanup() {
  if [[ "${KEEP_ARTIFACT_WORKTREE:-0}" != "1" && -d "$worktree" ]]; then
    git worktree remove --force "$worktree" >/dev/null 2>&1 || rm -rf "$worktree"
  fi
}
trap cleanup EXIT

mkdir -p "$(dirname "$worktree")"
rm -rf "$worktree"

git fetch "$remote" "$branch"
git worktree add --detach "$worktree" "$remote/$branch"

artifact_dir="$worktree/$artifact_root/$bundle_name"
rm -rf "$artifact_dir"
mkdir -p "$(dirname "$artifact_dir")"
cp -a "$bundle_dir" "$artifact_dir"

metadata="$artifact_dir/artifact_branch_metadata.txt"
{
  echo "created_at=$(date -Is)"
  echo "source_repo=$REPO_ROOT"
  echo "source_branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "source_commit=$(git rev-parse HEAD)"
  echo "jobs=$*"
  echo "bundle_dir=$bundle_dir"
  echo "artifact_branch=$branch"
  echo "artifact_path=$artifact_root/$bundle_name"
} > "$metadata"

git -C "$worktree" add -f "$artifact_root/$bundle_name"

if git -C "$worktree" diff --cached --quiet -- "$artifact_root/$bundle_name"; then
  echo "No artifact changes to commit."
else
  git -C "$worktree" commit -m "Add pass16 debug bundle ${job_label}"
fi

git -C "$worktree" push "$remote" "HEAD:$branch"

echo
echo "Pushed artifact bundle:"
echo "  branch: $branch"
echo "  path:   $artifact_root/$bundle_name"
echo "  local:  $bundle_dir"
