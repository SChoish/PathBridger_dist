#!/usr/bin/env bash
# Wait until paper_s0_pbf_pathwt GPU trainers are gone, then start the N>=8
# GPU eval sidecar (does not kill trains; polls until idle).
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger}"
RUN_GROUP="${RUN_GROUP:-paper_s0_pbf_pathwt}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}_gpu_eval_large_n}"
TRAIN_LOGROOT="${TRAIN_LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}}"
WATCHER="${WATCHER:-$REPO/scripts/watch_gpu_eval_large_n_paper_s0_pbf_pathwt.sh}"
POLL_SEC="${POLL_SEC:-30}"
read -r -a GPUS <<< "${GPU_LIST:-0 1}"

mkdir -p "$LOGROOT"
cd "$REPO" || exit 1

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] wait_gpu_large_n: %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/waiter.log"; }

echo $$ >"$LOGROOT/waiter.pid"
log "START waiting for train exit (group=${RUN_GROUP}); then GPU large-N on GPUS=${GPUS[*]}"

train_alive() {
  pgrep -f "main.py --agent=configs/pbf/.*--run_group=${RUN_GROUP}" >/dev/null 2>&1
}

# Also treat incomplete wave2 (no params_1000000 on all 8) as still training
# only when a live main.py exists — do not block forever on missing ckpts.
rounds=0
while train_alive; do
  rounds=$((rounds + 1))
  if (( rounds % 10 == 1 )); then
    n=$(pgrep -cf "main.py --agent=configs/pbf/.*--run_group=${RUN_GROUP}" || true)
    log "trains still live n≈${n} (poll ${POLL_SEC}s)"
  fi
  sleep "$POLL_SEC"
done

# Brief settle: orch may exit a few seconds after last train.
sleep 15
if train_alive; then
  log "WARN train reappeared; keep waiting"
  while train_alive; do sleep "$POLL_SEC"; done
  sleep 15
fi

# Confirm GPUs look free-ish (do not hard-fail if other users hold memory).
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  | tee -a "$LOGROOT/waiter.log" || true

if [[ -f "$LOGROOT/STOP" ]]; then
  log "STOP present — not launching GPU watcher"
  exit 0
fi

# Avoid double-launch.
if [[ -f "$LOGROOT/watcher.pid" ]]; then
  old="$(cat "$LOGROOT/watcher.pid" 2>/dev/null || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    log "GPU watcher already live pid=${old} — exit"
    exit 0
  fi
fi

log "train idle — launching GPU large-N watcher"
GPU_LIST="${GPUS[*]}" AUTO_EXIT="${AUTO_EXIT:-1}" \
  nohup bash "$WATCHER" >>"$LOGROOT/nohup_watcher.out" 2>&1 &
echo $! >"$LOGROOT/watcher_boot.pid"
log "launched watcher boot_pid=$! → see $LOGROOT/watcher.log"
# Mark train-gate done for operators.
touch "$TRAIN_LOGROOT/QUEUE_DONE" 2>/dev/null || true
touch "$LOGROOT/WAITER_HANDED_OFF"
log "HANDED_OFF"
