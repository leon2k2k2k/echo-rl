#!/usr/bin/env bash
set -euo pipefail

# Submit comparable terminal-agent smoke jobs for policy-update ablation.
# Defaults to plain baseline and ECHO-only; set INCLUDE_NEXTLAT=1 to add NextLat.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export PARTITION="${PARTITION:-h01}"
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
  export RUN_KIND="$run_kind"
  export CONFIG_PATH="$config_path"
  export RUN_ID="${RUN_KIND}-smoke-${SLURM_JOB_ID:-submit}-$(date +%Y%m%d-%H%M%S)"
  local job_id
  job_id="$(
    sbatch --parsable \
      -p "$PARTITION" \
      --output="$ECHO_LOG_DIR/%x-%j.out" \
      --error="$ECHO_LOG_DIR/%x-%j.err" \
      slurm/echo_nextlat_smoke.sbatch
  )"
  echo "$run_kind $job_id $config_path"
}

echo "Submitting ECHO ablation smokes on partition=$PARTITION"
echo "Format: RUN_KIND JOB_ID CONFIG_PATH"

submit_one plain-base echo_configs/qwen3_8b_rl_plain_smoke.yaml
submit_one echo-base echo_configs/qwen3_8b_rl_echo_smoke.yaml

if [[ "${INCLUDE_NEXTLAT:-0}" == "1" ]]; then
  submit_one echo-nextlat echo_configs/qwen3_8b_rl_nextlat_smoke.yaml
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
