#!/usr/bin/env bash
set -euo pipefail

# Self-contained ubuntu:22.04 seed pass@16 launcher.
#
# This script intentionally does not read TERMINAL_AGENT_TRAIN_PARQUET,
# TERMINAL_AGENT_VAL_PARQUET, or TERMINAL_AGENT_DATASET_LABEL from the caller.
# It pins those values after loading cluster_env.sh so stale smoke-data exports
# cannot leak into a baseline job.

usage() {
  cat <<'EOF'
Usage:
  bash scripts/submit_ubuntu22_pass16_fixed.sh [options]

Options:
  --only plain|echo|both       Which baseline(s) to submit. Default: both
  --plain-node NODE            Node for the plain baseline. Default: g06
  --echo-node NODE             Node for the ECHO baseline. Default: g08
  --node NODE                  Use the same node for both baselines.
  --partition PARTITION        Slurm partition. Default: a01
  --gres GRES                  Slurm gres request. Default: gpu:4
  --mem MEM                    Slurm memory request. Default: 64G
  --cpus N                     Slurm cpus-per-task. Default: 16
  --time TIME                  Slurm time limit. Default: 16:00:00
  --exclusive                  Add --exclusive to sbatch.
  --dry-run                    Print what would be submitted without sbatch.
  -h, --help                   Show this help.

Example:
  bash scripts/submit_ubuntu22_pass16_fixed.sh \
    --mem 64G \
    --cpus 16
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

only="both"
partition="a01"
plain_node="g06"
echo_node="g08"
gres="gpu:4"
mem="64G"
cpus="16"
time_limit="16:00:00"
exclusive=0
dry_run=0

while (($#)); do
  case "$1" in
    --only)
      only="${2:?missing value for --only}"
      shift 2
      ;;
    --plain-node)
      plain_node="${2:?missing value for --plain-node}"
      shift 2
      ;;
    --echo-node)
      echo_node="${2:?missing value for --echo-node}"
      shift 2
      ;;
    --node)
      plain_node="${2:?missing value for --node}"
      echo_node="$plain_node"
      shift 2
      ;;
    --partition)
      partition="${2:?missing value for --partition}"
      shift 2
      ;;
    --gres)
      gres="${2:?missing value for --gres}"
      shift 2
      ;;
    --mem)
      mem="${2:?missing value for --mem}"
      shift 2
      ;;
    --cpus)
      cpus="${2:?missing value for --cpus}"
      shift 2
      ;;
    --time)
      time_limit="${2:?missing value for --time}"
      shift 2
      ;;
    --exclusive)
      exclusive=1
      shift
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$only" in
  plain|echo|both) ;;
  *)
    echo "error: --only must be plain, echo, or both; got $only" >&2
    exit 2
    ;;
esac

# shellcheck source=/dev/null
source scripts/cluster_env.sh

dataset_root="$ECHO_RUNTIME_ROOT/data/ubuntu22_seed_echo"
train_parquet="$dataset_root/train.parquet"
val_parquet="$dataset_root/val.parquet"
log_dir="$ECHO_RUNTIME_ROOT/logs"
mkdir -p "$log_dir"

if [[ ! -f "$train_parquet" || ! -f "$val_parquet" ]]; then
  echo "Pinned ubuntu22 dataset is missing; building it now."
  bash scripts/build_ubuntu22_seed_echo_dataset.sh
fi

export TERMINAL_AGENT_TRAIN_PARQUET="$train_parquet"
export TERMINAL_AGENT_VAL_PARQUET="$val_parquet"
export TERMINAL_AGENT_DATASET_LABEL="ubuntu22_seed_echo"

check_terminal_agent_data_files

if [[ "$TERMINAL_AGENT_TRAIN_PARQUET" == */terminal_agent_smoke_train.parquet ]]; then
  echo "error: refusing to submit pass@16 on smoke train parquet" >&2
  exit 2
fi

export ECHO_HOLD_ON_FAILURE="1"
export ECHO_HOLD_ON_FAILURE_SECONDS="3600"
export ECHO_LOG_POLICY_TRAIN_METRICS="1"
export SKIP_ECHO_RUNTIME_IMPORT_CHECK="1"

stamp="$(date +%Y%m%d-%H%M%S)"

submit_one() {
  local tag="$1"
  local node="$2"
  local config_path="$3"
  local run_kind run_id

  run_kind="ubuntu22_seed_echo-${tag}"
  run_id="${run_kind}-${stamp}"

  export RUN_KIND="$run_kind"
  export RUN_ID="$run_id"
  export CONFIG_PATH="$config_path"

  local sbatch_args=(
    -p "$partition"
    --gres="$gres"
    --mem="$mem"
    --cpus-per-task="$cpus"
    --time="$time_limit"
    --output="$log_dir/%x-%j.out"
    --error="$log_dir/%x-%j.err"
    --parsable
  )
  sbatch_args+=(--nodelist="$node")
  if [[ "$exclusive" == "1" ]]; then
    sbatch_args+=(--exclusive)
  fi

  echo
  echo "Submitting $run_kind"
  echo "  partition=$partition"
  echo "  node=$node"
  echo "  gres=$gres"
  echo "  mem=$mem"
  echo "  cpus=$cpus"
  echo "  time=$time_limit"
  echo "  train=$TERMINAL_AGENT_TRAIN_PARQUET"
  echo "  val=$TERMINAL_AGENT_VAL_PARQUET"
  echo "  config=$CONFIG_PATH"

  if [[ "$dry_run" == "1" ]]; then
    echo "  dry_run=1"
    printf "  sbatch"
    printf " %q" "${sbatch_args[@]}" slurm/echo_nextlat_smoke.sbatch
    echo
    return 0
  fi

  local job_id
  job_id="$(sbatch "${sbatch_args[@]}" slurm/echo_nextlat_smoke.sbatch)"

  echo "  job=$job_id"
  echo "  out=$log_dir/echo-rl-smoke-$job_id.out"
  echo "  err=$log_dir/echo-rl-smoke-$job_id.err"
}

echo "Pinned ubuntu22 pass@16 launch"
echo "TERMINAL_AGENT_DATASET_LABEL=$TERMINAL_AGENT_DATASET_LABEL"
echo "TERMINAL_AGENT_TRAIN_PARQUET=$TERMINAL_AGENT_TRAIN_PARQUET"
echo "TERMINAL_AGENT_VAL_PARQUET=$TERMINAL_AGENT_VAL_PARQUET"
echo "LOG_DIR=$log_dir"

if [[ "$only" == "plain" || "$only" == "both" ]]; then
  submit_one \
    "plain-pass16-baseline" \
    "$plain_node" \
    "echo_configs/qwen3_8b_rl_plain_pass16_baseline.yaml"
fi

if [[ "$only" == "echo" || "$only" == "both" ]]; then
  submit_one \
    "echo-pass16-baseline" \
    "$echo_node" \
    "echo_configs/qwen3_8b_rl_echo_pass16_baseline.yaml"
fi

cat <<EOF

Monitor:
  cd $ECHO_REPO
  squeue -u "\${USER:-alex}" -o "%.18i %.10P %.30j %.8T %.10M %.20N %.30R"
  tail -f "$log_dir/echo-rl-smoke-JOBID.out" "$log_dir/echo-rl-smoke-JOBID.err"

Verify dataset in logs:
  grep -ahE "RUN_KIND=|CONFIG_PATH=|TERMINAL_AGENT_TRAIN_PARQUET=|TERMINAL_AGENT_VAL_PARQUET=|total_steps|dataset_max_rows" \\
    "$log_dir/echo-rl-smoke-JOBID.out" "$log_dir/echo-rl-smoke-JOBID.err"
EOF
