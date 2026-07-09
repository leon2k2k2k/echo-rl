#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/dump_pass16_debug_bundle.sh JOB_ID [JOB_ID ...]

Creates a compact debug bundle for pass@16 terminal-agent runs. The bundle is
safe to share back into chat: it includes score summaries, warning summaries,
metric excerpts, recent logs, and transcript final summaries when available.

Environment:
  BUNDLE_ROOT   Output parent. Default: $ECHO_RUNTIME_ROOT/debug_bundles
  TAIL_LINES    Recent raw log lines per job. Default: 1500

Example:
  bash scripts/dump_pass16_debug_bundle.sh 375559 375560
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -eq 0 ]]; then
  usage
  exit $([[ "$#" -eq 0 ]] && echo 2 || echo 0)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

tail_lines="${TAIL_LINES:-1500}"
stamp="$(date +%Y%m%d-%H%M%S)"
bundle_root="${BUNDLE_ROOT:-$ECHO_RUNTIME_ROOT/debug_bundles}"
job_label="$(printf '%s-' "$@" | sed 's/-$//')"
bundle_dir="$bundle_root/pass16-${job_label}-${stamp}"
mkdir -p "$bundle_dir/jobs"

echo "bundle_dir=$bundle_dir"

find_logs() {
  local job_id="$1"
  local candidates=(
    "$ECHO_RUNTIME_ROOT/logs/echo-rl-smoke-${job_id}.out"
    "$ECHO_RUNTIME_ROOT/logs/echo-rl-smoke-${job_id}.err"
    "$REPO_ROOT/logs/echo-rl-smoke-${job_id}.out"
    "$REPO_ROOT/logs/echo-rl-smoke-${job_id}.err"
  )
  local path
  for path in "${candidates[@]}"; do
    [[ -f "$path" ]] && printf '%s\n' "$path"
  done
}

extract_output_dir() {
  local logs=("$@")
  grep -ah '^OUTPUT_DIR=/' "${logs[@]}" 2>/dev/null | tail -1 | cut -d= -f2- || true
}

write_job_bundle() {
  local job_id="$1"
  local job_dir="$bundle_dir/jobs/$job_id"
  mkdir -p "$job_dir"

  mapfile -t logs < <(find_logs "$job_id")
  {
    echo "job_id=$job_id"
    echo "created_at=$(date -Is)"
    echo "repo=$REPO_ROOT"
    echo "runtime=$ECHO_RUNTIME_ROOT"
    echo
    echo "logs:"
    printf '  %s\n' "${logs[@]:-}"
  } > "$job_dir/metadata.txt"

  if (( ${#logs[@]} == 0 )); then
    echo "WARNING: no logs found for job $job_id" | tee "$job_dir/MISSING_LOGS.txt" >&2
    return 0
  fi

  cp "${logs[@]}" "$job_dir/" 2>/dev/null || true

  output_dir="$(extract_output_dir "${logs[@]}")"
  {
    grep -ahE \
      '^(RUN_KIND|RUN_ID|CONFIG_PATH|OUTPUT_DIR|TERMINAL_AGENT_DATASET_LABEL|TERMINAL_AGENT_TRAIN_PARQUET|TERMINAL_AGENT_VAL_PARQUET|ENTRYPOINT_START|ENTRYPOINT_DONE)=' \
      "${logs[@]}" 2>/dev/null || true
    echo
    echo "output_dir=$output_dir"
  } > "$job_dir/run_metadata.txt"

  grep -ah "WARNING" "${logs[@]}" > "$job_dir/warnings_raw.txt" 2>/dev/null || true
  grep -ahE \
    "Step [0-9]+:|reward/avg_pass_at_16|reward/avg_raw_reward|loss/avg_final_rewards|generate/verifier_error|generate/stop_reason|generate/parse_errors|generate/format_|timing/step|timing/generate|policy/final_loss|policy/policy_loss|policy/grad_norm|policy/loss_metrics/world_ce|policy/loss_metrics/world_tokens|policy/world_model_coeff|policy/nextlat_coeff" \
    "${logs[@]}" > "$job_dir/metric_lines.txt" 2>/dev/null || true
  grep -ahE \
    "rollout_done|verifier_done|verifier_timeout|verifier_error|agent_loop_done|prepare_batch_start|prepare_batch_done|Started: 'generate'|Finished: 'generate'" \
    "${logs[@]}" > "$job_dir/trajectory_events.txt" 2>/dev/null || true
  grep -ahE \
    "Traceback|RuntimeError|ERROR|FAILED|OutOfMemory|CUDA out of memory|Network is unreachable|No matching distribution|Image build/pull attempt" \
    "${logs[@]}" > "$job_dir/errors.txt" 2>/dev/null || true

  for log in "${logs[@]}"; do
    tail -n "$tail_lines" "$log" > "$job_dir/recent_$(basename "$log")" 2>/dev/null || true
  done

  if [[ -n "$output_dir" && -d "$output_dir" ]]; then
    {
      echo "output_dir=$output_dir"
      find "$output_dir" -maxdepth 2 -type f \
        \( -name 'terminal_transcripts.jsonl' -o -name 'heartbeat.log' -o -name 'final_snapshot.log' -o -name 'latest_ckpt_global_step.txt' \) \
        -printf '%s %p\n' 2>/dev/null | sort -n || true
    } > "$job_dir/output_files.txt"

    transcript="$output_dir/terminal_transcripts.jsonl"
    if [[ -f "$transcript" ]]; then
      "$PYTHON" scripts/summarize_terminal_transcripts.py "$transcript" --finals \
        > "$job_dir/transcript_finals.txt" 2> "$job_dir/transcript_finals.err" || true
      tail -n 200 "$transcript" > "$job_dir/transcript_tail.jsonl" 2>/dev/null || true
    fi

    if [[ -f "$output_dir/debug/heartbeat.log" ]]; then
      tail -n 300 "$output_dir/debug/heartbeat.log" > "$job_dir/heartbeat_tail.txt" 2>/dev/null || true
    fi
    if [[ -f "$output_dir/debug/final_snapshot.log" ]]; then
      tail -n 500 "$output_dir/debug/final_snapshot.log" > "$job_dir/final_snapshot_tail.txt" 2>/dev/null || true
    fi
  fi
}

jobs=("$@")

"$PYTHON" scripts/summarize_pass16_scores.py "${jobs[@]}" --steps --warnings 50 \
  > "$bundle_dir/score_summary.txt" 2> "$bundle_dir/score_summary.err" || true

{
  echo "created_at=$(date -Is)"
  echo "repo=$REPO_ROOT"
  echo "runtime=$ECHO_RUNTIME_ROOT"
  echo "jobs=${jobs[*]}"
  echo
  squeue -j "$(IFS=,; echo "${jobs[*]}")" -o "%.18i %.10P %.30j %.8T %.10M %.20N %.30R" 2>/dev/null || true
} > "$bundle_dir/manifest.txt"

for job in "${jobs[@]}"; do
  write_job_bundle "$job"
done

tar_path="${bundle_dir}.tgz"
tar -C "$bundle_root" -czf "$tar_path" "$(basename "$bundle_dir")"

echo
echo "Wrote bundle:"
echo "  $bundle_dir"
echo "  $tar_path"
echo
echo "Start with:"
echo "  cat \"$bundle_dir/score_summary.txt\""
echo "  find \"$bundle_dir\" -maxdepth 2 -type f | sort"
