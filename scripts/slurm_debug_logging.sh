#!/usr/bin/env bash

log_section() {
  echo
  echo "===== $* ====="
}

run_diag_cmd() {
  local label="$1"
  shift
  log_section "$label"
  "$@" || echo "[diag] command failed: $*"
}

print_redacted_env() {
  env | sort | grep -Ev '(TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE|CREDENTIAL|API_KEY|ACCESS_KEY)=' || true
}

print_slurm_preamble() {
  log_section "time/node/user"
  date -Is || true
  hostname || true
  id || true
  pwd || true

  log_section "slurm"
  env | sort | grep '^SLURM_' || true

  log_section "selected environment"
  print_redacted_env | grep -E '^(CONFIG_PATH|RUN_ID|RUN_KIND|OUTPUT_DIR|ECHO_|QWEN|TERMINAL_AGENT|SKYRL|PYTHON|PYTHONPATH|CUDA|NCCL|RAY|VLLM|HF_|TRANSFORMERS|TORCH|DOCKER|COMPOSE)' || true

  run_diag_cmd "limits" bash -lc 'ulimit -a'
  run_diag_cmd "disk" bash -lc 'df -h / /tmp /ram/tmp /container /home/fit/alex/WORK/leon /WORK/PUBLIC/alex_work/leon 2>/dev/null'
  run_diag_cmd "mount docker roots" bash -lc 'docker info 2>/dev/null | sed -n "/Docker Root Dir/p"; findmnt /container 2>/dev/null || true'
  run_diag_cmd "gpu" nvidia-smi
  run_diag_cmd "docker info" docker info
  run_diag_cmd "docker images smoke" bash -lc 'docker images --format "{{.Repository}}:{{.Tag}} {{.ID}} {{.Size}}" | grep -E "^(hb__tmax|ubuntu:22.04)" || true'
}

start_slurm_heartbeat() {
  local output_dir="$1"
  local interval="${ECHO_DEBUG_HEARTBEAT_INTERVAL:-60}"
  mkdir -p "$output_dir/debug"
  (
    while true; do
      {
        echo
        echo "===== heartbeat $(date -Is) ====="
        hostname || true
        ps -u "${USER:-alex}" -o pid,ppid,stat,etime,pcpu,pmem,args \
          | grep -E 'echo_rl|skyrl|raylet|gcs_server|VLLM|EngineCore|RayWorker|EchoFSDP|docker compose|python -m echo_rl' \
          | grep -v grep || true
        nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits || true
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits || true
        docker ps --format '{{.Names}} {{.Status}} {{.Image}}' | grep -E 'tmax|hb__|harbor' || true
        if [[ -e /tmp/ray/session_latest || -e /ram/tmp/ray/session_latest ]]; then
          local session
          session="$(readlink -f /tmp/ray/session_latest 2>/dev/null || readlink -f /ram/tmp/ray/session_latest 2>/dev/null || true)"
          echo "ray_session=$session"
          if [[ -n "$session" && -d "$session/logs" ]]; then
            find "$session/logs" -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -30 || true
          fi
        fi
      } >>"$output_dir/debug/heartbeat.log" 2>&1
      sleep "$interval"
    done
  ) &
  ECHO_HEARTBEAT_PID="$!"
  export ECHO_HEARTBEAT_PID
  echo "Started debug heartbeat pid=$ECHO_HEARTBEAT_PID interval=${interval}s log=$output_dir/debug/heartbeat.log"
}

stop_slurm_heartbeat() {
  if [[ -n "${ECHO_HEARTBEAT_PID:-}" ]]; then
    kill "$ECHO_HEARTBEAT_PID" 2>/dev/null || true
    wait "$ECHO_HEARTBEAT_PID" 2>/dev/null || true
  fi
}

collect_slurm_debug_artifacts() {
  local output_dir="$1"
  local exit_code="$2"
  mkdir -p "$output_dir/debug"

  {
    log_section "final status"
    echo "exit_code=$exit_code"
    date -Is || true
    hostname || true

    log_section "processes"
    ps -u "${USER:-alex}" -o pid,ppid,stat,etime,pcpu,pmem,args \
      | grep -E 'echo_rl|skyrl|raylet|gcs_server|VLLM|EngineCore|RayWorker|EchoFSDP|docker compose|python -m echo_rl' \
      | grep -v grep || true

    log_section "gpu final"
    nvidia-smi || true

    log_section "docker ps final"
    docker ps -a --format '{{.Names}} {{.Status}} {{.Image}}' | grep -E 'tmax|hb__|harbor' || true

    log_section "output dir files"
    find "$output_dir" -maxdepth 4 -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -200 || true
  } >>"$output_dir/debug/final_snapshot.log" 2>&1

  local session=""
  session="$(readlink -f /tmp/ray/session_latest 2>/dev/null || readlink -f /ram/tmp/ray/session_latest 2>/dev/null || true)"
  if [[ -n "$session" && -d "$session/logs" ]]; then
    echo "Collecting Ray logs from $session"
    {
      echo "ray_session=$session"
      find "$session/logs" -type f -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' 2>/dev/null | sort | tail -300 || true
    } >"$output_dir/debug/ray_log_index.txt" 2>&1
    tar -C "$session" -czf "$output_dir/debug/ray_logs.tgz" logs 2>"$output_dir/debug/ray_logs_tar.err" || true
  else
    echo "No Ray session found to collect"
  fi

  echo "Debug artifacts:"
  find "$output_dir/debug" -maxdepth 1 -type f -printf '  %s %p\n' 2>/dev/null | sort -n || true
}
