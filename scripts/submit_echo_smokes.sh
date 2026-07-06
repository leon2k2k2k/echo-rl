#!/usr/bin/env bash
set -euo pipefail

# Submit comparable ECHO-only and ECHO+NextLat smoke jobs.
# Run from the echo-rl repo on Alex's cluster.

# shellcheck source=/dev/null
source scripts/cluster_env.sh

export PARTITION="${PARTITION:-a01}"
export ECHO_LOG_DIR="${ECHO_LOG_DIR:-$ECHO_RUNTIME_ROOT/logs}"
mkdir -p "$ECHO_LOG_DIR"

ensure_terminal_agent_smoke_data
check_terminal_agent_data_files
"$PYTHON" scripts/check_echo_runtime_imports.py

export RUN_KIND=echo-base
export CONFIG_PATH=echo_configs/qwen3_8b_rl_echo_smoke.yaml
base_job="$(
  sbatch --parsable \
    -p "$PARTITION" \
    --output="$ECHO_LOG_DIR/%x-%j.out" \
    --error="$ECHO_LOG_DIR/%x-%j.err" \
    slurm/echo_nextlat_smoke.sbatch
)"
echo "Submitted $RUN_KIND job $base_job"

export RUN_KIND=echo-nextlat
export CONFIG_PATH=echo_configs/qwen3_8b_rl_nextlat_smoke.yaml
nextlat_job="$(
  sbatch --parsable \
    -p "$PARTITION" \
    --output="$ECHO_LOG_DIR/%x-%j.out" \
    --error="$ECHO_LOG_DIR/%x-%j.err" \
    slurm/echo_nextlat_smoke.sbatch
)"
echo "Submitted $RUN_KIND job $nextlat_job"

cat <<EOF

Monitor:
  squeue -u "\${USER:-alex}" -o "%.18i %.10P %.20j %.8T %.10M %.20R"
  bash scripts/tail_echo_smokes.sh

Metric grep:
  bash scripts/grep_smoke_metrics.sh
EOF
