#!/usr/bin/env bash
# PathBridger PBF paper params + temporal path weighting (seed0): all 8 pbf configs.
# Temporal-geodesic bridge weighting ON (beta=0.25, warm-up/ramp 100k). Paper PBF:
# endpoint=flow, h_a(action_chunk_horizon)=5 (puzzles too), batch=1024, 1M steps.
# Concurrency: GPUs × MAX_PER_GPU slots. Default 2 GPUs × 2 = 4 (cluster cap: 2 GPU/member).
# Fresh run_group; resumes in-place from latest params_*.pkl if a run already exists.
# Soft orch replace: ATTACH live trainers instead of double-launching.
# Error-21 thread pins + taskset required on Beyond-G (224-core).
#
# Path-weight / prefix knobs are CLI-pinned (do not rely on mutable code defaults):
#   --agent.path_weight_beta=0.25
#   --agent.path_weight_warmup=100000
#   --agent.path_weight_ramp=100000
#   --agent.prefix_model=deterministic
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
RUN_GROUP="${RUN_GROUP:-paper_s0_pbf_pathwt}"
SAVE_DIR="${SAVE_DIR:-$REPO/exp}"
EXP_ROOT="${EXP_ROOT:-$SAVE_DIR/pathbridger/${RUN_GROUP}}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/${RUN_GROUP}}"
MAX_PER_GPU="${MAX_PER_GPU:-2}"
# Override GPU set with e.g. GPU_LIST="2 3"; defaults to two GPUs.
read -r -a GPUS <<< "${GPU_LIST:-0 1}"
CPU_STRIDE="${CPU_STRIDE:-8}"

# Explicit agent overrides — filesystem defaults may change underfoot.
PATH_WEIGHT_BETA="${PATH_WEIGHT_BETA:-0.25}"
PATH_WEIGHT_WARMUP="${PATH_WEIGHT_WARMUP:-100000}"
PATH_WEIGHT_RAMP="${PATH_WEIGHT_RAMP:-100000}"
PREFIX_MODEL="${PREFIX_MODEL:-deterministic}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1
# Keep compilation serialized while several jobs start together, but leave GPU
# autotuning enabled (the XLA default) so H200 GEMMs get suitable algorithms.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1 ${XLA_FLAGS:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGROOT"
cd "$REPO"

# All configs under configs/pbf/
ENVS=(
  cube_single cube_double cube_triple
  puzzle_3x3 puzzle_4x4
  antmaze_medium antmaze_large scene
)

env_substr() {
  case "$1" in
    cube_single) echo 'cube-single-play' ;;
    cube_double) echo 'cube-double-play' ;;
    cube_triple) echo 'cube-triple-play' ;;
    puzzle_3x3) echo 'puzzle-3x3-play' ;;
    puzzle_4x4) echo 'puzzle-4x4-play' ;;
    antmaze_medium) echo 'antmaze-medium-navigate' ;;
    antmaze_large) echo 'antmaze-large-navigate' ;;
    scene) echo 'scene-play' ;;
    *) echo "$1" ;;
  esac
}

resolve_resume() {
  local env="$1"
  local es
  es="$(env_substr "$env")"
  "$PY" - "$EXP_ROOT" "$es" <<'PY'
import re, sys
from pathlib import Path
root, es = Path(sys.argv[1]), sys.argv[2]
if not root.is_dir():
    raise SystemExit(0)
cands = []
for d in root.iterdir():
    if not d.is_dir() or es not in d.name or '_flow_' not in d.name:
        continue
    ck = d / 'checkpoints'
    steps = []
    for p in ck.glob('params_*.pkl') if ck.is_dir() else []:
        m = re.search(r'params_(\d+)\.pkl$', p.name)
        if m:
            steps.append(int(m.group(1)))
    if not steps:
        continue
    cands.append((max(steps), d))
if not cands:
    raise SystemExit(0)
step, run = max(cands, key=lambda x: x[0])
print(f'{run}|{step}')
PY
}

find_live_train_pid() {
  local env="$1"
  local tag="pbf_pathwt_${env}_s0"
  local pidfile="$LOGROOT/${tag}.pid"
  local pid cmd
  if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
      if [[ "$cmd" == *"configs/pbf/${env}.py"* && "$cmd" == *"${RUN_GROUP}"* ]]; then
        echo "$pid"
        return 0
      fi
    fi
  fi
  # Fallback: match live main.py for this env + run_group.
  while read -r pid; do
    [[ -z "$pid" ]] && continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$cmd" == *"configs/pbf/${env}.py"* && "$cmd" == *"${RUN_GROUP}"* ]]; then
      echo "$pid"
      return 0
    fi
  done < <(pgrep -f "main.py --agent=configs/pbf/${env}\\.py" || true)
  return 1
}

JOBS=("${ENVS[@]}")

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/orchestrator.log"; }

# New LAUNCH/RESUME processes import from disk; require [B,5,D] bridge_targets trim.
preflight_bridge_targets_opt() {
  "$PY" - "$REPO" <<'PY'
from pathlib import Path
import sys
repo = Path(sys.argv[1])
agent = (repo / "agents" / "pathbridger.py").read_text()
data = (repo / "utils" / "datasets.py").read_text()
ok = (
    "bridge_targets" in agent
    and "bridge_targets must have shape" in agent
    and "'trajectory'" not in agent.split("_REQUIRED_BATCH_KEYS", 1)[-1][:800]
    and '"bridge_targets"' in data
    and "_BRIDGE_OFFSETS" in data
)
if not ok:
    raise SystemExit("preflight FAIL: PathBridger [B,5,D] bridge_targets opt missing on disk")
print("OK bridge_targets [B,5,D]")
PY
}

echo "$$" >"$LOGROOT/orchestrator.pid"
log "TRAIN queue ${#JOBS[@]} jobs; ${#GPUS[@]} GPUs × ${MAX_PER_GPU} slots; group=${RUN_GROUP} (pbf+pathwt)"
log "PIN agent.path_weight_beta=${PATH_WEIGHT_BETA} warmup=${PATH_WEIGHT_WARMUP} ramp=${PATH_WEIGHT_RAMP} prefix_model=${PREFIX_MODEL}"
if opt_msg="$(preflight_bridge_targets_opt)"; then
  log "OPT ${opt_msg} (applies to new LAUNCH/RESUME; ATTACH keeps in-memory code)"
else
  log "OPT preflight FAILED — refusing queue start"
  exit 1
fi

declare -a SLOT_PIDS
declare -a SLOT_TAGS
total_slots=$((${#GPUS[@]} * MAX_PER_GPU))
for ((i=0; i<total_slots; i++)); do
  SLOT_PIDS[$i]=""
  SLOT_TAGS[$i]=""
done

launch_one() {
  local slot="$1"
  local env="$2"
  local gpu_i=$((slot / MAX_PER_GPU))
  local gpu="${GPUS[$gpu_i]}"
  local slot_on_gpu=$((slot % MAX_PER_GPU))
  local cpu0=$(( (gpu_i * MAX_PER_GPU + slot_on_gpu) * CPU_STRIDE ))
  local cpu1=$((cpu0 + CPU_STRIDE - 1))
  local tag="pbf_pathwt_${env}_s0"
  local logf="$LOGROOT/${tag}.out"
  local resume_info restore_args=()
  local live_pid

  # Soft orch replace: claim already-running trainers instead of double-launch.
  live_pid="$(find_live_train_pid "$env" || true)"
  if [[ -n "${live_pid:-}" ]]; then
    log "ATTACH gpu=${gpu} cpus=${cpu0}-${cpu1} slot=${slot} ${tag} pid=${live_pid} (already running)"
    SLOT_PIDS[$slot]="$live_pid"
    SLOT_TAGS[$slot]="$tag"
    echo "$live_pid" >"$LOGROOT/${tag}.pid"
    return 0
  fi

  resume_info="$(resolve_resume "$env" || true)"
  if [[ -n "${resume_info:-}" ]]; then
    local run_dir restore_step
    IFS='|' read -r run_dir restore_step <<<"$resume_info"
    if (( restore_step >= 1000000 )); then
      log "SKIP STRICT_DONE ${tag} step=${restore_step} dir=$(basename "$run_dir")"
      SLOT_PIDS[$slot]=""
      SLOT_TAGS[$slot]=""
      return 0
    fi
    restore_args=(
      --restore_path="${run_dir}/checkpoints"
      --restore_step="${restore_step}"
      --run_dir="${run_dir}"
    )
    log "RESUME gpu=${gpu} cpus=${cpu0}-${cpu1} slot=${slot} ${tag} from step=${restore_step} code=bridge_targets[B,5,D]"
  else
    log "LAUNCH gpu=${gpu} cpus=${cpu0}-${cpu1} slot=${slot} ${tag} (fresh) code=bridge_targets[B,5,D]"
  fi

  if [[ -f "$logf" ]]; then
    mv -f "$logf" "${logf}.prev_$(TZ=Asia/Seoul date +%Y%m%d_%H%M%S)" || true
  fi

  # Drop stale bytecode so the new process cannot import an old trajectory path.
  rm -f "${REPO}/agents/__pycache__/pathbridger.cpython-"*.pyc \
        "${REPO}/utils/__pycache__/datasets.cpython-"*.pyc 2>/dev/null || true

  CUDA_VISIBLE_DEVICES="${gpu}" \
    nohup taskset -c "${cpu0}-${cpu1}" \
    "$PY" -u main.py \
      --agent="configs/pbf/${env}.py" \
      --seed=0 \
      --run_group="${RUN_GROUP}" \
      --save_dir="${SAVE_DIR}" \
      --train_steps=1000000 \
      --eval_interval=0 \
      --save_interval=100000 \
      --use_tqdm=False \
      --use_wandb=False \
      --async_prefetch=True \
      --agent.path_weight_beta="${PATH_WEIGHT_BETA}" \
      --agent.path_weight_warmup="${PATH_WEIGHT_WARMUP}" \
      --agent.path_weight_ramp="${PATH_WEIGHT_RAMP}" \
      --agent.prefix_model="${PREFIX_MODEL}" \
      "${restore_args[@]}" \
      >"$logf" 2>&1 &
  SLOT_PIDS[$slot]=$!
  SLOT_TAGS[$slot]="$tag"
  echo "${SLOT_PIDS[$slot]}" >"$LOGROOT/${tag}.pid"
  sleep 25
}

slots_busy() {
  local slot pid
  for ((slot=0; slot<total_slots; slot++)); do
    pid="${SLOT_PIDS[$slot]:-}"
    [[ -n "$pid" ]] && return 0
  done
  return 1
}

next_job=0
while (( next_job < ${#JOBS[@]} )) || slots_busy; do
  for ((slot=0; slot<total_slots; slot++)); do
    pid="${SLOT_PIDS[$slot]:-}"
    if [[ -n "$pid" ]]; then
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        log "DONE ${SLOT_TAGS[$slot]} pid=${pid}"
        SLOT_PIDS[$slot]=""
        SLOT_TAGS[$slot]=""
      fi
    fi
    if [[ -z "${SLOT_PIDS[$slot]:-}" && next_job -lt ${#JOBS[@]} ]]; then
      launch_one "$slot" "${JOBS[$next_job]}"
      next_job=$((next_job + 1))
    fi
  done
  sleep 30
done

touch "$LOGROOT/QUEUE_DONE"
log "QUEUE_DONE ${#JOBS[@]} jobs"
