#!/usr/bin/env bash
set -euo pipefail

# Prepare the Alex-cluster checkout for ECHO-only and ECHO+NextLat smoke jobs.
# This validates that the repo has the NextLat code, checks the SkyRL base commit,
# and syncs the ECHO package/configs into SkyRL.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck source=/dev/null
source scripts/cluster_env.sh

EXPECTED_SKYRL_COMMIT="43aab09782953cc7cfc93bda52b1635d717ce446"

echo "ECHO_REPO=$ECHO_REPO"
echo "SKYRL_DIR=$SKYRL_DIR"
echo "ECHO_RUNTIME_ROOT=$ECHO_RUNTIME_ROOT"
echo "PYTHON=$PYTHON"

if ! grep -R "nextlat_coeff" -n echo_rl configs patches >/dev/null; then
  echo "error: this echo-rl checkout does not contain the NextLat implementation." >&2
  echo "       Fetch/check out leon2k2k2k/echo-rl-nextlat first." >&2
  exit 1
fi

required_files=(
  configs/qwen3_8b_rl_echo_smoke.yaml
  configs/qwen3_8b_rl_nextlat_smoke.yaml
  scripts/cluster_env.sh
  scripts/sync_echo_to_skyrl.sh
  scripts/submit_echo_smokes.sh
  scripts/grep_smoke_metrics.sh
  slurm/echo_nextlat_smoke.sbatch
)
for path in "${required_files[@]}"; do
  if [[ ! -f "$path" ]]; then
    echo "error: missing required smoke file: $path" >&2
    exit 1
  fi
done

if [[ ! -d "$SKYRL_DIR/.git" ]]; then
  echo "error: SKYRL_DIR is not a git checkout: $SKYRL_DIR" >&2
  echo "       Clone SkyRL and check out $EXPECTED_SKYRL_COMMIT first." >&2
  exit 1
fi

actual_skyrl_commit="$(git -C "$SKYRL_DIR" rev-parse HEAD)"
if [[ "$actual_skyrl_commit" != "$EXPECTED_SKYRL_COMMIT" ]]; then
  echo "error: SkyRL is at $actual_skyrl_commit, expected $EXPECTED_SKYRL_COMMIT" >&2
  echo "       Run:" >&2
  echo "         cd $SKYRL_DIR" >&2
  echo "         git fetch --depth 1 origin $EXPECTED_SKYRL_COMMIT" >&2
  echo "         git checkout $EXPECTED_SKYRL_COMMIT" >&2
  exit 1
fi

bash -n \
  scripts/cluster_env.sh \
  scripts/sync_echo_to_skyrl.sh \
  scripts/submit_echo_smokes.sh \
  scripts/grep_smoke_metrics.sh \
  slurm/echo_nextlat_smoke.sbatch

bash scripts/sync_echo_to_skyrl.sh

cat <<EOF

Ready to submit:
  cd $ECHO_REPO
  export PARTITION=\${PARTITION:-a01}
  bash scripts/submit_echo_smokes.sh

After submission, use the printed job IDs:
  squeue -j "JOB1,JOB2" -o "%.18i %.10P %.20j %.8T %.10M %.20R"
  sacct -j "JOB1,JOB2" --format=JobID,JobName%24,State,ExitCode,Elapsed
  bash scripts/grep_smoke_metrics.sh JOB1
  bash scripts/grep_smoke_metrics.sh JOB2
EOF
