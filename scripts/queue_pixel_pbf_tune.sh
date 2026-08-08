#!/usr/bin/env bash
# Full-data offline pixel PBF: 8 gap anchors, then 32 checkpoint-only N/T evaluations.
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger_dist_pixel_pbf}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
SAVE_DIR="${SAVE_DIR:-/raid/ext_csv/PathBridger_dist_pixel_pbf/exp}"
LOGROOT="${LOGROOT:-/raid/ext_csv/PathBridger_dist_pixel_pbf/nohup_logs/pixel_pbf_full_offline_tune}"
DATASET_DIR="${DATASET_DIR:-/raid/ext_csv/datasets/ogbench_visual}"
# shellcheck disable=SC2206
GPU_LIST=(${GPU_LIST:-0 1})
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
RUN_GROUP="${RUN_GROUP:-pixel_pbf_full_gap_nt_s0}"
OFFLINE_STEPS="${OFFLINE_STEPS:-1000000}"
EVAL_EPISODES="${EVAL_EPISODES:-50}"
AUTO_RESUME="${AUTO_RESUME:-1}"
SEED="${SEED:-0}"
POLL_SEC="${POLL_SEC:-20}"
CPU_STRIDE="${CPU_STRIDE:-16}"
CPU_BASE0="${CPU_BASE0:-0}"
LAUNCH_STAGGER_SEC="${LAUNCH_STAGGER_SEC:-15}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.40}"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export EIGEN_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1 ${XLA_FLAGS:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}" WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$LOGROOT/results"
cd "$REPO" || exit 1
SOURCE_COMMIT=$(git rev-parse HEAD)

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/orchestrator.log"; }
tag_num() { printf '%s' "$1" | tr '.' 'p' | tr '-' 'm'; }

if [[ -f "$LOGROOT/STOP" ]]; then
  log "FATAL stale STOP marker: $LOGROOT/STOP"
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  log "FATAL dataset_dir missing: $DATASET_DIR"
  exit 1
fi

ENVS=(
  visual-antmaze-large-navigate-v0
  visual-cube-double-play-v0
  visual-puzzle-4x4-play-v0
  visual-scene-play-v0
)
GAPS=(5 10)
NT_SEARCH=("1|0" "2|0.25" "16|0.5" "32|1.0")

declare -a SLOT_GPUS=() CPU_BASES=()
slot_i=0
for gpu in "${GPU_LIST[@]}"; do
  for ((slot = 0; slot < SLOTS_PER_GPU; slot++)); do
    SLOT_GPUS+=("$gpu")
    CPU_BASES+=($((CPU_BASE0 + slot_i * CPU_STRIDE)))
    slot_i=$((slot_i + 1))
  done
done
MAX=${#SLOT_GPUS[@]}

declare -a PIDS=() KEYS=() GPU_SLOTS=()
STOP_SENT=0
PHASE_FAILED=0

next_slot() {
  local slot used active
  for ((slot = 0; slot < MAX; slot++)); do
    used=0
    for active in "${GPU_SLOTS[@]:-}"; do
      [[ "$active" == "$slot" ]] && used=1 && break
    done
    ((used == 0)) && { echo "$slot"; return 0; }
  done
  return 1
}

reap_one() {
  local i pid key rc
  for i in "${!PIDS[@]}"; do
    pid=${PIDS[$i]}
    if ! kill -0 "$pid" 2>/dev/null; then
      key=${KEYS[$i]}
      wait "$pid" 2>/dev/null && rc=0 || rc=$?
      log "DONE key=$key pid=$pid rc=$rc"
      if [[ -f "$LOGROOT/STOP" ]]; then
        touch "$LOGROOT/$key.STOPPED"
      elif ((rc == 0)); then
        touch "$LOGROOT/$key.DONE"
      else
        printf 'rc=%s\n' "$rc" >"$LOGROOT/$key.FAIL"
        PHASE_FAILED=1
      fi
      rm -f "$LOGROOT/$key.pid" "$LOGROOT/$key.LOCK"
      unset 'PIDS[i]' 'KEYS[i]' 'GPU_SLOTS[i]'
      PIDS=("${PIDS[@]}") KEYS=("${KEYS[@]}") GPU_SLOTS=("${GPU_SLOTS[@]}")
      return 0
    fi
  done
  return 1
}

clean_stop() {
  local pid
  ((STOP_SENT == 1)) && return
  STOP_SENT=1
  log "STOP requested; sending one clean TERM to live process groups"
  for pid in "${PIDS[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap 'touch "$LOGROOT/STOP"; clean_stop' INT TERM

source_is_fixed() {
  [[ "$(git rev-parse HEAD)" == "$SOURCE_COMMIT" ]] && git diff --quiet
}

find_checkpoint() {
  local env="$1" gap="$2" root checkpoint
  root="$SAVE_DIR/pixel_o2o/${RUN_GROUP}_gap${gap}"
  checkpoint=$(find "$root" -type f \
    -path "*/pixel_pbf_${env}_sd$(printf '%03d' "$SEED")_*/checkpoints/step_0.pkl" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || return 1
  printf '%s\n' "$checkpoint"
}

find_resume_checkpoint() {
  local env="$1" gap="$2" root checkpoint
  root="$SAVE_DIR/pixel_o2o/${RUN_GROUP}_gap${gap}"
  checkpoint=$(find "$root" -type f \
    -path "*/pixel_pbf_${env}_sd$(printf '%03d' "$SEED")_*/checkpoints/offline_step_*.pkl" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || return 1
  printf '%s\n' "$checkpoint"
}

launch_job() {
  local phase="$1" env="$2" gap="$3" n="${4:-1}" temp="${5:-0}"
  local key slot gpu cpu0 cpu1 out cfg group checkpoint output pid
  local -a cmd
  [[ ! -f "$LOGROOT/STOP" ]] || return 2
  source_is_fixed || { log "FATAL tracked source changed during sweep"; return 1; }
  if [[ "$phase" == train ]]; then
    key="train_${env}_gap${gap}_s${SEED}"
  else
    key="eval_${env}_gap${gap}_N${n}_T$(tag_num "$temp")_s${SEED}"
  fi
  key=$(printf '%s' "$key" | tr '-' '_')
  [[ ! -f "$LOGROOT/$key.DONE" ]] || { log "SKIP done $key"; return 0; }

  while ((${#PIDS[@]} >= MAX)); do
    [[ ! -f "$LOGROOT/STOP" ]] || return 2
    reap_one || sleep 5
  done
  [[ ! -f "$LOGROOT/STOP" ]] || return 2
  slot=$(next_slot) || return 1
  gpu=${SLOT_GPUS[$slot]}
  cpu0=${CPU_BASES[$slot]}
  cpu1=$((cpu0 + CPU_STRIDE - 1))
  out="$LOGROOT/$key.out"

  if [[ "$phase" == train ]]; then
    group="${RUN_GROUP}_gap${gap}"
    cfg=$(printf '{"endpoint_value_scale":%s,"eval_num_candidates":1,"eval_temperature":0.0,"tune_tag":"full_offline_gap%s"}' "$gap" "$gap")
    cmd=("$PY" -u train_pixel.py
      --algorithm=pixel_pbf --env_name="$env" --seed="$SEED"
      --run_group="$group" --protocol_suite=pixel_pbf_full_offline_gap
      --dataset_dir="$DATASET_DIR" --save_dir="$SAVE_DIR"
      --offline_steps="$OFFLINE_STEPS" --online_steps=0 --random_steps=0
      --eval_steps=0 --eval_episodes=0 --frame_stack=3
      --resume_interval=0 --nouse_tqdm --config_json="$cfg")
    if [[ "$AUTO_RESUME" == 1 ]] && checkpoint=$(find_resume_checkpoint "$env" "$gap"); then
      log "RESUME env=$env gap=$gap checkpoint=$checkpoint"
      cmd+=(--restore_path="$checkpoint" --restore_step=-1 --resume_in_place)
    fi
  else
    checkpoint=$(find_checkpoint "$env" "$gap") || {
      log "FATAL missing checkpoint env=$env gap=$gap"; return 1;
    }
    output="$LOGROOT/results/$key.json"
    cmd=("$PY" -u evaluate_pixel.py
      --checkpoint="$checkpoint" --episodes="$EVAL_EPISODES" --seed="$SEED"
      --num_candidates="$n" --endpoint_temperature="$temp"
      --output_path="$output")
  fi

  log "LAUNCH phase=$phase gpu=$gpu cpus=$cpu0-$cpu1 key=$key"
  export CUDA_VISIBLE_DEVICES="$gpu"
  setsid taskset -c "$cpu0-$cpu1" "${cmd[@]}" >"$out" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$LOGROOT/$key.pid"
  printf '%s\n' "$pid" >"$LOGROOT/$key.LOCK"
  PIDS+=("$pid") KEYS+=("$key") GPU_SLOTS+=("$slot")
  ((LAUNCH_STAGGER_SEC > 0)) && sleep "$LAUNCH_STAGGER_SEC"
}

finish_phase() {
  while ((${#PIDS[@]} > 0)); do
    [[ -f "$LOGROOT/STOP" ]] && clean_stop
    reap_one || sleep "$POLL_SEC"
  done
  [[ ! -f "$LOGROOT/STOP" && "$PHASE_FAILED" -eq 0 ]]
}

printf '%s\n' $$ >"$LOGROOT/orchestrator.pid"
log "START commit=$SOURCE_COMMIT full_offline_jobs=8 eval_jobs=32 steps=$OFFLINE_STEPS"
log "PHASE_START full_offline_train"
for env in "${ENVS[@]}"; do
  for gap in "${GAPS[@]}"; do
    launch_job train "$env" "$gap" || break 2
  done
done

if finish_phase; then
  log "PHASE_START nt_evaluation"
  for env in "${ENVS[@]}"; do
    for gap in "${GAPS[@]}"; do
      for cell in "${NT_SEARCH[@]}"; do
        IFS='|' read -r n temp <<<"$cell"
        launch_job eval "$env" "$gap" "$n" "$temp" || break 3
      done
    done
  done
  finish_phase
fi
rc=$?
if ((rc == 0)); then log "QUEUE_COMPLETE"; else log "QUEUE_STOPPED_OR_FAILED rc=$rc"; fi
rm -f "$LOGROOT/orchestrator.pid"
log "EXIT"
exit "$rc"
