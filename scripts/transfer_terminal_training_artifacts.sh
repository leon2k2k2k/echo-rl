#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/transfer_terminal_training_artifacts.sh JOB_ID [LABEL]

Copies a terminal-agent training run's Slurm logs and debug artifacts into
training_artifacts/, commits them on an artifact branch, and pushes.

Environment:
  REMOTE=private                         Git remote to fetch/push.
  BRANCH=data-terminal-artifacts         Artifact branch to update.
  LOG_DIR=logs                           Slurm log directory.

Examples:
  scripts/transfer_terminal_training_artifacts.sh 375314 nextlat-c04-live
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

job_id="${1:-}"
label="${2:-}"

if [[ -z "$job_id" ]]; then
  usage >&2
  exit 2
fi

if ! [[ "$job_id" =~ ^[0-9]+$ ]]; then
  echo "ERROR: JOB_ID must be numeric, got: $job_id" >&2
  exit 2
fi

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

remote="${REMOTE:-private}"
branch="${BRANCH:-data-terminal-artifacts}"
log_dir="${LOG_DIR:-logs}"
out_log="$log_dir/echo-rl-smoke-${job_id}.out"
err_log="$log_dir/echo-rl-smoke-${job_id}.err"

if [[ ! -f "$out_log" && ! -f "$err_log" ]]; then
  echo "ERROR: no logs found for job $job_id under $log_dir" >&2
  exit 1
fi

combined_logs=()
[[ -f "$out_log" ]] && combined_logs+=("$out_log")
[[ -f "$err_log" ]] && combined_logs+=("$err_log")

output_dir="$(
  grep -ah '^OUTPUT_DIR=' "${combined_logs[@]}" 2>/dev/null \
    | tail -1 \
    | cut -d= -f2-
)"

run_kind="$(
  grep -ah '^RUN_KIND=' "${combined_logs[@]}" 2>/dev/null \
    | tail -1 \
    | cut -d= -f2-
)"

run_id="$(
  grep -ah '^RUN_ID=' "${combined_logs[@]}" 2>/dev/null \
    | tail -1 \
    | cut -d= -f2-
)"

if [[ -z "$output_dir" ]]; then
  echo "ERROR: could not find OUTPUT_DIR in logs for job $job_id" >&2
  exit 1
fi

if [[ ! -d "$output_dir" ]]; then
  echo "ERROR: OUTPUT_DIR does not exist: $output_dir" >&2
  exit 1
fi

if [[ -z "$label" ]]; then
  label="${run_kind:-${run_id:-terminal-training}}"
fi

safe_label="$(
  printf '%s' "$label" \
    | tr '[:upper:]' '[:lower:]' \
    | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//'
)"

if [[ -z "$safe_label" ]]; then
  safe_label="terminal-training"
fi

artifact_dir="training_artifacts/${safe_label}-${job_id}"

echo "JOB_ID=$job_id"
echo "RUN_KIND=${run_kind:-unknown}"
echo "RUN_ID=${run_id:-unknown}"
echo "OUTPUT_DIR=$output_dir"
echo "ARTIFACT_DIR=$artifact_dir"
echo "BRANCH=$branch"
echo "REMOTE=$remote"

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "ERROR: working tree has local changes; commit/stash before switching artifact branch" >&2
  exit 1
fi

git fetch "$remote" "$branch"
git switch "$branch"
git pull --ff-only "$remote" "$branch"

rm -rf "$artifact_dir"
mkdir -p "$artifact_dir/slurm_logs" "$artifact_dir/debug" "$artifact_dir/policy_rank_logs"

for log_file in "${combined_logs[@]}"; do
  cp "$log_file" "$artifact_dir/slurm_logs/"
done

if [[ -f "$output_dir/terminal_transcripts.jsonl" ]]; then
  cp "$output_dir/terminal_transcripts.jsonl" "$artifact_dir/"
else
  echo "WARNING: missing terminal_transcripts.jsonl" >&2
fi

for debug_file in final_snapshot.log heartbeat.log ray_log_index.txt ray_logs_tar.err; do
  if [[ -f "$output_dir/debug/$debug_file" ]]; then
    cp "$output_dir/debug/$debug_file" "$artifact_dir/debug/"
  fi
done

if [[ -d "$output_dir/policy_rank_logs" ]]; then
  find "$output_dir/policy_rank_logs" -maxdepth 1 -type f -name 'rank_*.log' -print0 \
    | while IFS= read -r -d '' rank_log; do
        cp "$rank_log" "$artifact_dir/policy_rank_logs/"
      done
fi

if [[ -f "$output_dir/ckpts/latest_ckpt_global_step.txt" ]]; then
  mkdir -p "$artifact_dir/ckpts"
  cp "$output_dir/ckpts/latest_ckpt_global_step.txt" "$artifact_dir/ckpts/"
fi

metadata="$artifact_dir/metadata.txt"
{
  echo "job_id=$job_id"
  echo "run_kind=${run_kind:-}"
  echo "run_id=${run_id:-}"
  echo "output_dir=$output_dir"
  echo "created_at=$(date -Is)"
  echo
  grep -ahE \
    '^(CONFIG_PATH|RUN_KIND|RUN_ID|OUTPUT_DIR|CONFIG_OVERRIDES|ENTRYPOINT_START|ENTRYPOINT_DONE)=' \
    "${combined_logs[@]}" 2>/dev/null || true
  echo
  grep -ahE \
    '^CONFIG_OVERRIDE_ARG\[[0-9]+\]=' \
    "${combined_logs[@]}" 2>/dev/null || true
} > "$metadata"

find "$artifact_dir" -type f ! -name '*.gz' -print0 \
  | xargs -0 -r gzip -f

git add -f "$artifact_dir"

if git diff --cached --quiet -- "$artifact_dir"; then
  echo "No artifact changes to commit."
else
  git commit -m "Transfer ${safe_label} training artifacts for job ${job_id}"
fi

git push "$remote" "$branch"

echo
echo "Pushed $branch to $remote."
echo "Artifact files:"
find "$artifact_dir" -type f | sort
