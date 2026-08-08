#!/usr/bin/env bash
# Best-NT endpoint fixed; sweep bridge-flow (M × prefix-T), T≠0.
# Default: M∈{1,4,8,16} × pt∈{0.25,0.5,1.0}, 25 ep/task, GPU.
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger_dist}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_GROUP="${RUN_GROUP:-paper_s0_pbf_jflow}"
EXP_ROOT="${EXP_ROOT:-$REPO/exp/pathbridger/${RUN_GROUP}}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}_mt_grid}"
# shellcheck disable=SC2206
MS=(${PREFIX_MS:-1 4 8 16})
# shellcheck disable=SC2206
PTS=(${PREFIX_TS:-0.25 0.5 1.0})
# shellcheck disable=SC2206
GPU_LIST=(${GPU_LIST:-0 1})
MAX="${MAX:-${#GPU_LIST[@]}}"
EPISODES="${EPISODES:-25}"
POLL_SEC="${POLL_SEC:-20}"
EVAL_GRID="${EVAL_GRID:-best}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGROOT"
cd "$REPO" || exit 1

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/sweep.log"; }

pt_tok() {
  "$PY" -c "t=float('$1'); print(str(int(t)) if float(t)==int(t) else str(t).replace('.','p'))"
}

declare -a PIDS=()
declare -a KEYS=()
declare -a GPU_SLOTS=()

reap_one() {
  local i pid key rc lock
  for i in "${!PIDS[@]}"; do
    pid="${PIDS[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      key="${KEYS[$i]}"
      wait "$pid" 2>/dev/null && rc=0 || rc=$?
      log "DONE key=${key} pid=${pid} rc=${rc}"
      local rname="${key%%_step*}"
      local rest="${key#*_step}"
      local step="${rest%%_m*}"
      local mid="${rest#*_m}"
      local m="${mid%%_pt*}"
      local ptok="${mid#*_pt}"
      lock="${EXP_ROOT}/${rname}/cpu_eval/step_${step}_m${m}_pt${ptok}.LOCK"
      if [[ -f "$lock" ]] && [[ "$(cat "$lock" 2>/dev/null)" == "$pid" ]]; then
        rm -f "$lock"
      fi
      unset 'PIDS[i]' 'KEYS[i]' 'GPU_SLOTS[i]'
      PIDS=("${PIDS[@]}")
      KEYS=("${KEYS[@]}")
      GPU_SLOTS=("${GPU_SLOTS[@]}")
      return 0
    fi
  done
  return 1
}

next_gpu_slot() {
  local used s
  for ((s=0; s<MAX; s++)); do
    used=0
    for u in "${GPU_SLOTS[@]:-}"; do
      [[ "$u" == "$s" ]] && used=1 && break
    done
    if (( used == 0 )); then
      echo "$s"
      return 0
    fi
  done
  return 1
}

lock_held() {
  local lock="$1" pid
  [[ -f "$lock" ]] || return 1
  pid="$(cat "$lock" 2>/dev/null || true)"
  [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null
}

launch_one() {
  local run_dir="$1" step="$2" m="$3" pt="$4"
  local name out donef lock slot gpu ptok
  name="$(basename "$run_dir")"
  if [[ "$name" == _* || ! -f "$run_dir/checkpoints/params_${step}.pkl" ]]; then
    return 0
  fi
  ptok="$(pt_tok "$pt")"
  mkdir -p "${run_dir}/cpu_eval"
  donef="${run_dir}/cpu_eval/step_${step}_m${m}_pt${ptok}.DONE"
  lock="${run_dir}/cpu_eval/step_${step}_m${m}_pt${ptok}.LOCK"
  if [[ -f "$donef" ]]; then
    return 0
  fi
  if lock_held "$lock"; then
    return 0
  fi
  rm -f "$lock" 2>/dev/null || true

  while (( ${#PIDS[@]} >= MAX )); do
    if ! reap_one; then sleep 3; fi
  done
  slot="$(next_gpu_slot)" || slot=0
  gpu="${GPU_LIST[$slot]}"
  out="$LOGROOT/${name}_step${step}_m${m}_pt${ptok}.out"
  log "LAUNCH gpu=${gpu} ${name} step=${step} m=${m} pt=${pt} ep=${EPISODES}"
  (
    set +e
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "$PY" -u scripts/gpu_eval_pbf_best_m.py \
        --gpu="${gpu}" \
        --run-dir="${run_dir}" \
        --checkpoint-step="${step}" \
        --episodes="${EPISODES}" \
        --seed=0 \
        --grid="${EVAL_GRID}" \
        --prefix-samples="${m}" \
        --prefix-temperature="${pt}" \
      >"$out" 2>&1
    exit $?
  ) &
  local child=$!
  echo "$child" >"$lock"
  PIDS+=("$child")
  KEYS+=("${name}_step${step}_m${m}_pt${ptok}")
  GPU_SLOTS+=("$slot")
}

list_pending() {
  "$PY" - "$EXP_ROOT" "${MS[*]}" "${PTS[*]}" <<'PY' || true
from pathlib import Path
import os, sys

def tok(t):
    t = float(t)
    return str(int(t)) if t == int(t) else str(t).replace('.', 'p')

root = Path(sys.argv[1])
ms = [int(x) for x in sys.argv[2].split()]
pts = [float(x) for x in sys.argv[3].split()]
steps = (800000, 900000, 1000000)
if not root.is_dir():
    raise SystemExit
for d in sorted(root.iterdir()):
    if not d.is_dir() or d.name.startswith('_') or not (d / 'flags.json').exists():
        continue
    ck = d / 'checkpoints'
    for step in steps:
        if not (ck / f'params_{step}.pkl').is_file():
            continue
        for m in ms:
            for pt in pts:
                ptok = tok(pt)
                done = d / 'cpu_eval' / f'step_{step}_m{m}_pt{ptok}.DONE'
                if done.is_file():
                    continue
                lock = d / 'cpu_eval' / f'step_{step}_m{m}_pt{ptok}.LOCK'
                if lock.is_file():
                    try:
                        pid = int(lock.read_text().strip())
                    except Exception:
                        pid = -1
                    if pid > 0:
                        try:
                            os.kill(pid, 0)
                            continue
                        except OSError:
                            pass
                print(f'{d}|{step}|{m}|{pt}')
PY
}

log "START MT-grid group=${RUN_GROUP} Ms=${MS[*]} PTs=${PTS[*]} GPUs=${GPU_LIST[*]} MAX=${MAX} ep=${EPISODES}"
echo $$ >"$LOGROOT/sweep.pid"

while true; do
  if [[ -f "$LOGROOT/STOP" ]]; then
    log "STOP requested"
    break
  fi
  while reap_one; do :; done
  mapfile -t pending < <(list_pending)
  if (( ${#pending[@]} > 0 )); then
    for item in "${pending[@]}"; do
      [[ -z "$item" ]] && continue
      IFS='|' read -r rd st m pt <<<"$item"
      launch_one "$rd" "$st" "$m" "$pt"
    done
  else
    if (( ${#PIDS[@]} == 0 )); then
      log "SWEEP_COMPLETE M×pt grid"
      break
    fi
  fi
  sleep "$POLL_SEC"
done

while (( ${#PIDS[@]} > 0 )); do
  if ! reap_one; then sleep 5; fi
done
log "EXIT"
rm -f "$LOGROOT/sweep.pid"
