#!/usr/bin/env bash
set -euo pipefail

# Submit comparable ECHO-only and ECHO+NextLat smoke jobs.
# Run from the echo-rl repo on Alex's cluster.

export PARTITION="${PARTITION:-a01}"

export RUN_KIND=echo-base
export CONFIG_PATH=echo_configs/qwen3_8b_rl_echo_smoke.yaml
base_job="$(sbatch --parsable -p "$PARTITION" slurm/echo_nextlat_smoke.sbatch)"
echo "Submitted $RUN_KIND job $base_job"

export RUN_KIND=echo-nextlat
export CONFIG_PATH=echo_configs/qwen3_8b_rl_nextlat_smoke.yaml
nextlat_job="$(sbatch --parsable -p "$PARTITION" slurm/echo_nextlat_smoke.sbatch)"
echo "Submitted $RUN_KIND job $nextlat_job"

cat <<EOF

Monitor:
  squeue -j "$base_job,$nextlat_job" -o "%.18i %.10P %.20j %.8T %.10M %.20R"
  sacct -j "$base_job,$nextlat_job" --format=JobID,JobName%24,State,ExitCode,Elapsed
  tail -f logs/echo-rl-smoke-$base_job.out logs/echo-rl-smoke-$nextlat_job.out

Metric grep:
  bash scripts/grep_smoke_metrics.sh "$base_job"
  bash scripts/grep_smoke_metrics.sh "$nextlat_job"
EOF
