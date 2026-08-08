#!/usr/bin/env bash
# After paper_s0_pbf_pathwt GPU large-N eval drains, launch stochastic bridge
# trains (path weighting OFF) for both backends, sequentially:
#   1) low_rank_gaussian → paper_s0_pbf_lrg
#   2) joint_flow        → paper_s0_pbf_jflow
# Each backend: train queue + CPU eval keepalive + wait→GPU large-N sidecar.
# Does not stop pathwt CPU eval (CPU-only). Does not start while pathwt GPU
# eval workers are still alive.
#
# Compare target: provenance PBF (deterministic), not pathwt.
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
CHAIN_LOGROOT="${CHAIN_LOGROOT:-$REPO/nohup_logs/paper_s0_pbf_stoch_chain}"
POLL_SEC="${POLL_SEC:-60}"
read -r -a GPUS <<< "${GPU_LIST:-0 1}"

mkdir -p "$CHAIN_LOGROOT"
cd "$REPO" || exit 1

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] stoch_chain: %s\n' "$(ts)" "$*" | tee -a "$CHAIN_LOGROOT/chain.log"; }

echo $$ >"$CHAIN_LOGROOT/chain.pid"
log "START wait pathwt GPU large-N idle, then lrg → jflow (path_weight_beta=0)"

pathwt_gpu_eval_alive() {
  pgrep -f 'gpu_eval_pbf_pathwt_large_n\.py' >/dev/null 2>&1 \
    || pgrep -f 'watch_gpu_eval_large_n_paper_s0_pbf_pathwt\.sh' >/dev/null 2>&1
}

rounds=0
while pathwt_gpu_eval_alive; do
  rounds=$((rounds + 1))
  if (( rounds % 5 == 1 )); then
    n=$(pgrep -cf 'gpu_eval_pbf_pathwt_large_n\.py' || true)
    log "pathwt GPU large-N still live n≈${n} (poll ${POLL_SEC}s)"
  fi
  if [[ -f "$CHAIN_LOGROOT/STOP" ]]; then
    log "STOP — exit before launch"
    exit 0
  fi
  sleep "$POLL_SEC"
done

sleep 20
if pathwt_gpu_eval_alive; then
  log "WARN pathwt GPU reappeared; keep waiting"
  while pathwt_gpu_eval_alive; do sleep "$POLL_SEC"; done
  sleep 20
fi

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  | tee -a "$CHAIN_LOGROOT/chain.log" || true

if [[ -f "$CHAIN_LOGROOT/STOP" ]]; then
  log "STOP — exit before launch"
  exit 0
fi

launch_backend() {
  local pm="$1"
  local rg
  case "$pm" in
    low_rank_gaussian) rg=paper_s0_pbf_lrg ;;
    joint_flow) rg=paper_s0_pbf_jflow ;;
    *) rg="paper_s0_pbf_${pm}" ;;
  esac
  local train_log="$REPO/nohup_logs/${rg}"
  local cpu_log="$REPO/nohup_logs/${rg}_cpu_eval"
  local gpu_log="$REPO/nohup_logs/${rg}_gpu_eval_large_n"
  mkdir -p "$train_log" "$cpu_log" "$gpu_log"

  log "LAUNCH train backend=${pm} group=${rg}"
  PREFIX_MODEL="$pm" RUN_GROUP="$rg" PATH_WEIGHT_BETA=0 \
    GPU_LIST="${GPUS[*]}" \
    nohup bash "$REPO/scripts/queue_paper_s0_pbf_stoch.sh" \
    >>"$train_log/nohup_orch.out" 2>&1 &
  echo $! >"$train_log/orch_boot.pid"
  log "train orch boot_pid=$! → $train_log/orchestrator.log"

  # CPU eval keepalive (stoch argv — do not fight pathwt keepalive).
  if [[ ! -f "$cpu_log/EVAL_QUEUE_DONE" ]]; then
    RUN_GROUP="$rg" LOGROOT="$cpu_log" \
      nohup bash "$REPO/scripts/keepalive_watch_cpu_eval_paper_s0_pbf_stoch.sh" \
      >>"$cpu_log/nohup_keepalive.out" 2>&1 &
    echo $! >"$cpu_log/keepalive_boot.pid"
    log "cpu keepalive boot_pid=$! group=${rg}"
  else
    log "cpu EVAL_QUEUE_DONE already — skip keepalive ${rg}"
  fi

  # GPU large-N after this backend's trains exit (stoch watcher wrapper).
  RUN_GROUP="$rg" TRAIN_LOGROOT="$train_log" LOGROOT="$gpu_log" \
    WATCHER="$REPO/scripts/watch_gpu_eval_large_n_paper_s0_pbf_stoch.sh" \
    GPU_LIST="${GPUS[*]}" AUTO_EXIT=1 \
    nohup bash "$REPO/scripts/wait_train_then_gpu_eval_large_n_paper_s0_pbf_pathwt.sh" \
    >>"$gpu_log/nohup_waiter.out" 2>&1 &
  echo $! >"$gpu_log/waiter_boot.pid"
  log "gpu large-N waiter boot_pid=$! group=${rg}"

  # Block until train orchestrator exits (QUEUE_DONE or orch gone).
  local orch_pid
  orch_pid="$(cat "$train_log/orchestrator.pid" 2>/dev/null || true)"
  # Wait a bit for orch to write pid.
  for _ in 1 2 3 4 5 6; do
    orch_pid="$(cat "$train_log/orchestrator.pid" 2>/dev/null || true)"
    [[ -n "${orch_pid:-}" ]] && break
    sleep 5
  done
  if [[ -n "${orch_pid:-}" ]]; then
    log "wait train orch pid=${orch_pid} group=${rg}"
    while kill -0 "$orch_pid" 2>/dev/null; do
      if [[ -f "$CHAIN_LOGROOT/STOP" ]]; then
        log "STOP during ${rg} train — exit (trains left running)"
        exit 0
      fi
      sleep "$POLL_SEC"
    done
  else
    log "WARN no orchestrator.pid for ${rg}; poll QUEUE_DONE / main.py"
    while [[ ! -f "$train_log/QUEUE_DONE" ]] \
      && pgrep -f "main.py --agent=configs/pbf/.*--run_group=${rg}" >/dev/null 2>&1; do
      sleep "$POLL_SEC"
    done
  fi
  touch "$CHAIN_LOGROOT/${rg}_TRAIN_ORCH_DONE"
  log "TRAIN orch done group=${rg}"
}

launch_backend low_rank_gaussian
launch_backend joint_flow

touch "$CHAIN_LOGROOT/CHAIN_DONE"
log "CHAIN_DONE both backends queued through train orch exit (evals may still drain)"
