#!/usr/bin/env bash
set -euo pipefail

# Submit comparable terminal-agent smoke jobs for policy-update ablation.
# Defaults to plain baseline and ECHO-only; set INCLUDE_NEXTLAT=1 to add NextLat.
#
# vLLM inference servers use fixed ports in these smoke configs. Running two
# smokes on the same node can make their routers/servers collide before training
# starts, so the default submits one job to h01 and one job to a01. Set
# SUBMIT_MODE=sequential to run all jobs on one partition with afterany deps, or
# SUBMIT_MODE=parallel only when node isolation is guaranteed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export PARTITION="${PARTITION:-h01}"
export PLAIN_PARTITION="${PLAIN_PARTITION:-h01}"
export ECHO_PARTITION="${ECHO_PARTITION:-a01}"
export NEXTLAT_PARTITION="${NEXTLAT_PARTITION:-$ECHO_PARTITION}"
export SUBMIT_MODE="${SUBMIT_MODE:-split-partitions}"
export ECHO_LOG_DIR="${ECHO_LOG_DIR:-$ECHO_RUNTIME_ROOT/logs}"
export ECHO_LOG_POLICY_TRAIN_METRICS="${ECHO_LOG_POLICY_TRAIN_METRICS:-1}"
mkdir -p "$ECHO_LOG_DIR"

ensure_terminal_agent_smoke_data
check_terminal_agent_data_files
if [[ "${SKIP_ECHO_RUNTIME_IMPORT_CHECK:-0}" != "1" ]]; then
  "$PYTHON" scripts/check_echo_runtime_imports.py
else
  echo "Skipping ECHO runtime import check because SKIP_ECHO_RUNTIME_IMPORT_CHECK=1"
fi

submit_one() {
  local run_kind="$1"
  local config_path="$2"
  local partition="$3"
  local dependency="${4:-}"
  export RUN_KIND="$run_kind"
  export CONFIG_PATH="$config_path"
  export RUN_ID="${RUN_KIND}-smoke-submit-$(date +%Y%m%d-%H%M%S)"
  local sbatch_args=(
    -p "$partition"
    --output="$ECHO_LOG_DIR/%x-%j.out"
    --error="$ECHO_LOG_DIR/%x-%j.err"
  )
  if [[ -n "$dependency" ]]; then
    sbatch_args+=(--dependency="afterany:$dependency")
  fi
  if [[ "${SBATCH_EXCLUSIVE:-0}" == "1" ]]; then
    sbatch_args+=(--exclusive)
  fi
  local job_id
  job_id="$(
    sbatch --parsable "${sbatch_args[@]}" slurm/echo_nextlat_smoke.sbatch
  )"
  SUBMITTED_JOB_ID="$job_id"
  echo "$run_kind $job_id partition=$partition dependency=${dependency:-none} $config_path"
}

case "$SUBMIT_MODE" in
  split-partitions|sequential|parallel) ;;
  *)
    echo "error: SUBMIT_MODE must be split-partitions, sequential, or parallel; got $SUBMIT_MODE" >&2
    exit 2
    ;;
esac

if [[ "$SUBMIT_MODE" == "parallel" && "${SBATCH_EXCLUSIVE:-0}" != "1" ]]; then
  cat >&2 <<'EOF'
warning: SUBMIT_MODE=parallel can collide fixed vLLM ports if Slurm places jobs
on the same node. Prefer SUBMIT_MODE=split-partitions or set SBATCH_EXCLUSIVE=1.
EOF
fi

echo "Submitting ECHO ablation smokes with SUBMIT_MODE=$SUBMIT_MODE"
echo "Format: RUN_KIND JOB_ID PARTITION DEPENDENCY CONFIG_PATH"

dependency=""
if [[ "$SUBMIT_MODE" == "split-partitions" ]]; then
  submit_one plain-base echo_configs/qwen3_8b_rl_plain_smoke.yaml "$PLAIN_PARTITION"
  submit_one echo-base echo_configs/qwen3_8b_rl_echo_smoke.yaml "$ECHO_PARTITION"
elif [[ "$SUBMIT_MODE" == "sequential" ]]; then
  submit_one plain-base echo_configs/qwen3_8b_rl_plain_smoke.yaml "$PARTITION" "$dependency"
  dependency="$SUBMITTED_JOB_ID"
  submit_one echo-base echo_configs/qwen3_8b_rl_echo_smoke.yaml "$PARTITION" "$dependency"
  dependency="$SUBMITTED_JOB_ID"
else
  submit_one plain-base echo_configs/qwen3_8b_rl_plain_smoke.yaml "$PARTITION"
  submit_one echo-base echo_configs/qwen3_8b_rl_echo_smoke.yaml "$PARTITION"
fi

if [[ "${INCLUDE_NEXTLAT:-0}" == "1" ]]; then
  if [[ "$SUBMIT_MODE" == "split-partitions" ]]; then
    submit_one echo-nextlat echo_configs/qwen3_8b_rl_nextlat_smoke.yaml "$NEXTLAT_PARTITION" "$SUBMITTED_JOB_ID"
  elif [[ "$SUBMIT_MODE" == "sequential" ]]; then
    submit_one echo-nextlat echo_configs/qwen3_8b_rl_nextlat_smoke.yaml "$PARTITION" "$dependency"
  else
    submit_one echo-nextlat echo_configs/qwen3_8b_rl_nextlat_smoke.yaml "$PARTITION"
  fi
fi

cat <<EOF

Monitor:
  cd $ECHO_REPO
  squeue -u "\${USER:-alex}" -o "%.18i %.10P %.20j %.8T %.10M %.20R"
  bash scripts/tail_echo_smokes.sh

Metric grep:
  cd $ECHO_REPO
  bash scripts/grep_smoke_metrics.sh

Failure diagnosis after a job exits:
  cd $ECHO_REPO
  D=\$(ls -td "$ECHO_RUNTIME_ROOT"/outputs/*smoke-* | head -1)
  python3 scripts/diagnose_echo_smoke_failure.py "\$D" --tail 120
EOF
