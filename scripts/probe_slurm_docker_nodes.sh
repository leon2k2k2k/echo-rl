#!/usr/bin/env bash
set -uo pipefail

PARTITIONS="${PARTITIONS:-a01 h01}"
NODES="${NODES:-}"
GRES="${GRES:-gpu:4}"
SRUN_IMMEDIATE="${SRUN_IMMEDIATE:-30}"
OUTER_TIMEOUT="${OUTER_TIMEOUT:-50}"
DOCKER_TIMEOUT="${DOCKER_TIMEOUT:-10}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-8G}"
TIME_LIMIT="${TIME_LIMIT:-00:03:00}"
NODE_STATE_RE="${NODE_STATE_RE:-idle|mix|alloc}"

expand_nodes() {
  local expr="$1"
  if command -v scontrol >/dev/null 2>&1; then
    scontrol show hostnames "$expr"
  else
    echo "$expr"
  fi
}

candidate_nodes() {
  local partition="$1"
  if [[ -n "$NODES" ]]; then
    echo "$NODES" | tr ' ,' '\n' | awk 'NF'
    return 0
  fi
  sinfo -h -p "$partition" -o "%N %T" \
    | awk -v re="$NODE_STATE_RE" '$2 ~ re {print $1}' \
    | while read -r expr; do expand_nodes "$expr"; done \
    | sort -u
}

probe_node() {
  local partition="$1"
  local node="$2"
  local output rc status docker_status gpu_status gpu_count containers root_dir server_version

  output="$(
    timeout "${OUTER_TIMEOUT}s" srun -p "$partition" -w "$node" -N1 -n1 \
      --gres="$GRES" \
      --cpus-per-task="$CPUS_PER_TASK" \
      --mem="$MEM" \
      --time="$TIME_LIMIT" \
      --immediate="$SRUN_IMMEDIATE" \
      bash -lc '
        set +e
        echo "host=$(hostname)"
        echo "docker_bin=$(command -v docker || true)"
        docker_out=$(timeout '"$DOCKER_TIMEOUT"'s docker info 2>&1)
        docker_rc=$?
        echo "docker_rc=$docker_rc"
        printf "%s\n" "$docker_out" | grep -E "Server Version|Docker Root Dir|permission denied|Cannot connect|Is the docker daemon running|error|Error" || true
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        echo "gpu_count=$gpu_count"
        echo "containers=$(docker ps -q 2>/dev/null | wc -l)"
        docker image inspect ubuntu:22.04 >/dev/null 2>&1
        echo "ubuntu_cached=$?"
        task_images=$(docker image ls "hb__tmax/*" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | wc -l)
        echo "task_images=$task_images"
      ' 2>&1
  )"
  rc=$?

  if [[ "$rc" == "124" ]]; then
    status="TIMEOUT"
  elif grep -qiE "Unable to allocate resources|Job allocation .* has been revoked|Could not allocate|temporarily disabled|Requested node configuration is not available|immediate" <<<"$output"; then
    status="NO_ALLOC"
  else
    docker_status="BAD"
    grep -q "docker_rc=0" <<<"$output" && docker_status="OK"
    gpu_count="$(grep -m1 '^gpu_count=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    containers="$(grep -m1 '^containers=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    ubuntu_cached="$(grep -m1 '^ubuntu_cached=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    task_images="$(grep -m1 '^task_images=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    server_version="$(grep -m1 'Server Version:' <<<"$output" | sed 's/^[[:space:]]*//')"
    root_dir="$(grep -m1 'Docker Root Dir:' <<<"$output" | sed 's/^[[:space:]]*//')"
    if [[ "${gpu_count:-0}" =~ ^[0-9]+$ && "${gpu_count:-0}" -ge 4 ]]; then
      gpu_status="OK"
    else
      gpu_status="BAD"
    fi
    if [[ "$docker_status" == "OK" && "$gpu_status" == "OK" ]]; then
      status="OK"
    elif [[ "$docker_status" != "OK" ]]; then
      status="DOCKER_BAD"
    else
      status="GPU_BAD"
    fi
  fi

  if [[ "${ubuntu_cached:-?}" == "0" ]]; then
    ubuntu_cached="yes"
  elif [[ "${ubuntu_cached:-?}" =~ ^[0-9]+$ ]]; then
    ubuntu_cached="no"
  fi

  printf "%-4s %-6s %-11s gpu=%-2s containers=%-3s ubuntu=%-3s task_images=%-3s %s %s\n" \
    "$partition" "$node" "$status" "${gpu_count:-?}" "${containers:-?}" \
    "${ubuntu_cached:-?}" "${task_images:-?}" "${server_version:-}" "${root_dir:-}"

  if [[ "${VERBOSE:-0}" == "1" && "$status" != "OK" ]]; then
    sed 's/^/  | /' <<<"$output"
  fi
}

for partition in $PARTITIONS; do
  echo "===== $partition candidates ====="
  nodes="$(candidate_nodes "$partition")"
  if [[ -z "$nodes" ]]; then
    echo "no candidate nodes"
    continue
  fi
  echo "$nodes" | tr '\n' ' '
  echo
  for node in $nodes; do
    probe_node "$partition" "$node"
  done
done
