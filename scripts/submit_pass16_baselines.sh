#!/usr/bin/env bash
set -euo pipefail

# Submit the plain-RL vs ECHO-only pass@16 baseline pair.
# Each step uses one prompt group:
#   train_batch_size=1, n_samples_per_prompt=16, agent_max_concurrency=8.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export TERMINAL_AGENT_TRAIN_PARQUET="${TERMINAL_AGENT_TRAIN_PARQUET:-${ECHO_PROBE_TRAIN_PARQUET:-$ECHO_RUNTIME_ROOT/data/tmax_echo/train.parquet}}"
export TERMINAL_AGENT_VAL_PARQUET="${TERMINAL_AGENT_VAL_PARQUET:-${ECHO_PROBE_VAL_PARQUET:-$ECHO_RUNTIME_ROOT/data/tmax_echo/val.parquet}}"
check_terminal_agent_data_files

export ECHO_HOLD_ON_FAILURE="${ECHO_HOLD_ON_FAILURE:-1}"
export ECHO_HOLD_ON_FAILURE_SECONDS="${ECHO_HOLD_ON_FAILURE_SECONDS:-3600}"
export ECHO_LOG_POLICY_TRAIN_METRICS="${ECHO_LOG_POLICY_TRAIN_METRICS:-1}"

plain_partition="${PLAIN_PARTITION:-a01}"
echo_partition="${ECHO_PARTITION:-a01}"
plain_node="${PLAIN_NODE:-}"
echo_node="${ECHO_NODE:-}"
time_limit="${SBATCH_TIME:-16:00:00}"
gres="${SBATCH_GRES:-gpu:4}"
stamp="$(date +%Y%m%d-%H%M%S)"

submit_one() {
  local tag="$1"
  local partition="$2"
  local node="$3"
  local config_path="$4"

  export RUN_KIND="$tag"
  export RUN_ID="${RUN_KIND}-${stamp}"
  export CONFIG_PATH="$config_path"

  echo
  echo "Submitting $RUN_KIND"
  echo "PARTITION=$partition"
  if [[ -n "$node" ]]; then
    echo "NODE=$node"
  fi
  echo "GRES=$gres"
  echo "RUN_ID=$RUN_ID"
  echo "CONFIG_PATH=$CONFIG_PATH"

  local sbatch_args=(
    -p "$partition"
    --gres="$gres"
    --time="$time_limit"
    --parsable
  )
  if [[ -n "$node" ]]; then
    sbatch_args+=(--nodelist="$node")
  fi

  local job_id
  job_id="$(
    sbatch \
      "${sbatch_args[@]}" \
      slurm/echo_nextlat_smoke.sbatch
  )"

  echo "$tag JOB=$job_id"
  echo "  logs/echo-rl-smoke-$job_id.out"
  echo "  logs/echo-rl-smoke-$job_id.err"
}

echo "Submitting pass@16 baselines"
echo "TERMINAL_AGENT_TRAIN_PARQUET=$TERMINAL_AGENT_TRAIN_PARQUET"
echo "TERMINAL_AGENT_VAL_PARQUET=$TERMINAL_AGENT_VAL_PARQUET"
echo "ECHO_HOLD_ON_FAILURE=$ECHO_HOLD_ON_FAILURE"
echo "ECHO_HOLD_ON_FAILURE_SECONDS=$ECHO_HOLD_ON_FAILURE_SECONDS"

submit_one "plain-pass16-baseline" "$plain_partition" "$plain_node" "echo_configs/qwen3_8b_rl_plain_pass16_baseline.yaml"
submit_one "echo-pass16-baseline" "$echo_partition" "$echo_node" "echo_configs/qwen3_8b_rl_echo_pass16_baseline.yaml"

cat <<'EOF'

Monitor:
  squeue -u "${USER:-alex}" -o "%.18i %.10P %.20j %.8T %.10M %.20R"

Key progress:
  grep -ahE "RUN_KIND=|RUN_ID=|CONFIG_PATH=|generate_start|batch_num_seq:|batch_padded_seq_len|forward_backward_start|micro_batches|Step [0-9]+:|avg_final_rewards|avg_pass_at_16|avg_pass_at_8|Training done|Traceback|RuntimeError|OutOfMemory|FAILED" \
    logs/echo-rl-smoke-*.err logs/echo-rl-smoke-*.out \
    | tail -240

Per-job tail:
  tail -f logs/echo-rl-smoke-JOBID.err logs/echo-rl-smoke-JOBID.out
EOF
