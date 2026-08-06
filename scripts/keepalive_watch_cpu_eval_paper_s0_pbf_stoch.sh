#!/usr/bin/env bash
# Keep-alive for stochastic PBF CPU eval (separate argv from pathwt keepalive).
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger_dist}"
RUN_GROUP="${RUN_GROUP:-paper_s0_pbf_lrg}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}_cpu_eval}"
WATCHER="${WATCHER:-$REPO/scripts/watch_cpu_eval_paper_s0_pbf_stoch.sh}"
BACKOFF_SEC="${BACKOFF_SEC:-15}"

mkdir -p "$LOGROOT"
ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] keepalive: %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/keepalive.log"; }

echo $$ >"$LOGROOT/keepalive.pid"
log "START supervising ${WATCHER} RUN_GROUP=${RUN_GROUP} (pid=$$)"

child_watcher=""

term_foreign_watchers() {
  local pid cmd
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    [[ "$pid" == "$$" ]] && continue
    [[ -n "$child_watcher" && "$pid" == "$child_watcher" ]] && continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    [[ "$cmd" == *keepalive_watch_cpu_eval* ]] && continue
    [[ "$cmd" == *watch_cpu_eval_paper_s0_pbf_stoch.sh* ]] || continue
    # Only kill stoch watchers that look like our RUN_GROUP (env in /proc).
    if [[ -r "/proc/$pid/environ" ]] && \
       tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | grep -qx "RUN_GROUP=${RUN_GROUP}"; then
      log "TERM foreign stoch watcher pid=${pid} group=${RUN_GROUP}"
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done < <(pgrep -f 'watch_cpu_eval_paper_s0_pbf_stoch\.sh' || true)
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
  RUN_GROUP="$RUN_GROUP" LOGROOT="$LOGROOT" EXP_ROOT="${EXP_ROOT:-}" \
    bash "$WATCHER" >>"$LOGROOT/watcher_nohup.out" 2>&1 &
  child_watcher=$!
  echo "$child_watcher" >"$LOGROOT/watcher.pid"
  wait "$child_watcher"
  rc=$?
  child_watcher=""
  log "watcher exited rc=${rc}; backoff ${BACKOFF_SEC}s"
  sleep "$BACKOFF_SEC"
done
