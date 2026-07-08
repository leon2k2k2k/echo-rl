#!/usr/bin/env bash
set -uo pipefail

# Find nodes that can start the Echo-RL terminal-agent training shape and can
# access Docker from inside the Slurm allocation.
#
# Defaults match the 4-GPU baseline jobs closely enough to test real
# schedulability while releasing the allocation immediately after diagnostics.

PARTITIONS="${PARTITIONS:-a01 h01}"
NODES="${NODES:-}"
GRES="${GRES:-gpu:4}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-256G}"
TIME_LIMIT="${TIME_LIMIT:-05:00:00}"
SRUN_IMMEDIATE="${SRUN_IMMEDIATE:-10}"
OUTER_TIMEOUT="${OUTER_TIMEOUT:-90}"
DOCKER_TIMEOUT="${DOCKER_TIMEOUT:-10}"
MIN_GPU_COUNT="${MIN_GPU_COUNT:-4}"
MIN_GPU_MEM_MB="${MIN_GPU_MEM_MB:-79000}"
NODE_STATE_RE="${NODE_STATE_RE:-idle|mix}"
VERBOSE="${VERBOSE:-0}"

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
  local output rc status docker_status gpu_status gpu_count containers
  local min_mem ubuntu_cached task_images root_dir server_version

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
        echo "id=$(id)"
        echo "docker_bin=$(command -v docker || true)"
        docker_out=$(timeout '"$DOCKER_TIMEOUT"'s docker info 2>&1)
        docker_rc=$?
        echo "docker_rc=$docker_rc"
        printf "%s\n" "$docker_out" | grep -E "Server Version|Docker Root Dir|permission denied|Cannot connect|Is the docker daemon running|error|Error" || true
        gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
        echo "gpu_count=$gpu_count"
        min_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk "NR==1 || \$1 < min {min=\$1} END {if (min == \"\") min=0; print min}")
        echo "min_gpu_mem_mb=$min_mem"
        echo "containers=$(docker ps -q 2>/dev/null | wc -l)"
        docker image inspect ubuntu:22.04 >/dev/null 2>&1
        echo "ubuntu_cached=$?"
        task_images=$(docker image ls "hb__tmax/*" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | wc -l)
        echo "task_images=$task_images"
        if [[ -S /var/run/docker.sock ]]; then
          stat -c "docker_sock=%A:%U:%G:%a" /var/run/docker.sock || true
        fi
      ' 2>&1
  )"
  rc=$?

  if [[ "$rc" == "124" ]]; then
    status="TIMEOUT"
  elif grep -qiE "Unable to allocate resources|Job allocation .* has been revoked|Could not allocate|temporarily disabled|Requested node configuration is not available|immediate|Priority" <<<"$output"; then
    status="NO_ALLOC"
  else
    docker_status="BAD"
    grep -q "docker_rc=0" <<<"$output" && docker_status="OK"
    gpu_count="$(grep -m1 '^gpu_count=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    min_mem="$(grep -m1 '^min_gpu_mem_mb=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    containers="$(grep -m1 '^containers=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    ubuntu_cached="$(grep -m1 '^ubuntu_cached=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    task_images="$(grep -m1 '^task_images=' <<<"$output" | cut -d= -f2 | tr -d ' ')"
    server_version="$(grep -m1 'Server Version:' <<<"$output" | sed 's/^[[:space:]]*//')"
    root_dir="$(grep -m1 'Docker Root Dir:' <<<"$output" | sed 's/^[[:space:]]*//')"

    gpu_status="BAD"
    if [[ "${gpu_count:-0}" =~ ^[0-9]+$ && "${gpu_count:-0}" -ge "$MIN_GPU_COUNT" &&
          "${min_mem:-0}" =~ ^[0-9]+$ && "${min_mem:-0}" -ge "$MIN_GPU_MEM_MB" ]]; then
      gpu_status="OK"
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

  printf "%-4s %-6s %-11s gpu=%-2s min_mem=%-6s containers=%-3s ubuntu=%-3s task_images=%-3s %s %s\n" \
    "$partition" "$node" "$status" "${gpu_count:-?}" "${min_mem:-?}" \
    "${containers:-?}" "${ubuntu_cached:-?}" "${task_images:-?}" \
    "${server_version:-}" "${root_dir:-}"

  if [[ "$status" == "OK" ]]; then
    FOUND_OK+=("$partition:$node")
  fi

  if [[ "$VERBOSE" == "1" && "$status" != "OK" ]]; then
    sed 's/^/  | /' <<<"$output"
  fi
}

FOUND_OK=()

echo "Probe shape:"
echo "  partitions=$PARTITIONS"
echo "  nodes=${NODES:-<auto from sinfo states $NODE_STATE_RE>}"
echo "  gres=$GRES cpus=$CPUS_PER_TASK mem=$MEM time=$TIME_LIMIT immediate=$SRUN_IMMEDIATE"
echo "  min_gpu_count=$MIN_GPU_COUNT min_gpu_mem_mb=$MIN_GPU_MEM_MB"
echo

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

echo
echo "===== usable nodes ====="
if ((${#FOUND_OK[@]} == 0)); then
  echo "none"
  exit 1
fi
printf "%s\n" "${FOUND_OK[@]}"
