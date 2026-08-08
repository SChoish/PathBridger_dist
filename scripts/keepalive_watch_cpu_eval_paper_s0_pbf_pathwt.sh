#!/usr/bin/env bash
# Keep-alive supervisor for paper_s0_pbf_pathwt CPU eval watcher.
# Restarts on unexpected exit until EVAL_QUEUE_DONE (or STOP file).
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger}"
RUN_GROUP="${RUN_GROUP:-paper_s0_pbf_pathwt}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}_cpu_eval}"
WATCHER="${WATCHER:-$REPO/scripts/watch_cpu_eval_paper_s0_pbf_pathwt.sh}"
BACKOFF_SEC="${BACKOFF_SEC:-15}"

mkdir -p "$LOGROOT"
ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] keepalive: %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/keepalive.log"; }

echo $$ >"$LOGROOT/keepalive.pid"
log "START supervising ${WATCHER} (pid=$$)"

child_watcher=""

term_foreign_watchers() {
  local pid cmd
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$$" ]] && continue
    [[ -n "$child_watcher" && "$pid" == "$child_watcher" ]] && continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$cmd" == *keepalive_watch_cpu_eval* ]] && continue
    [[ "$cmd" == *watch_cpu_eval_paper_s0_pbf_pathwt.sh* ]] || continue
    log "TERM foreign watcher pid=${pid}"
    kill -TERM "$pid" 2>/dev/null || true
  done < <(pgrep -f 'watch_cpu_eval_paper_s0_pbf_pathwt\.sh' || true)
}

while true; do
  if [[ -f "$LOGROOT/STOP" ]]; then
    log "STOP file present — exit"
    [[ -n "$child_watcher" ]] && kill -TERM "$child_watcher" 2>/dev/null || true
    exit 0
  fi
  if [[ -f "$LOGROOT/EVAL_QUEUE_DONE" ]]; then
    log "EVAL_QUEUE_DONE — exit"
    exit 0
  fi

  term_foreign_watchers
  sleep 1

  log "launch watcher"
  EVAL_GRID="${EVAL_GRID:-best}" \
    bash "$WATCHER" >>"$LOGROOT/nohup_watcher.out" 2>&1 &
  child_watcher=$!
  echo "$child_watcher" >"$LOGROOT/watcher.pid"
  log "watcher pid=${child_watcher}"
  wait "$child_watcher"
  rc=$?
  child_watcher=""
  if [[ -f "$LOGROOT/EVAL_QUEUE_DONE" || -f "$LOGROOT/STOP" ]]; then
    log "watcher exited rc=${rc} (terminal) — stop supervise"
    exit 0
  fi
  log "watcher exited rc=${rc} — restart in ${BACKOFF_SEC}s"
  sleep "$BACKOFF_SEC"
done
