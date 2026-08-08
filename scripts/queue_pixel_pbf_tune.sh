#!/usr/bin/env bash
# Pixel PBF sweep: 8 shared offline anchors, then 32 online gap x (N,T) branches.
set +e
set -uo pipefail

REPO="${REPO:-/home/ext_csv/PathBridger_AF_pixel_lapo}"
PY="${PY:-/home/ext_csv/miniconda3/envs/offrl/bin/python}"
SAVE_DIR="${SAVE_DIR:-$REPO/exp}"
LOGROOT="${LOGROOT:-$REPO/nohup_logs/pixel_pbf_tune}"
DATASET_DIR="${DATASET_DIR:-/raid/ext_csv/datasets/ogbench_visual}"
# shellcheck disable=SC2206
GPU_LIST=(${GPU_LIST:-0 1})
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
RUN_GROUP="${RUN_GROUP:-pixel_pbf_gap_nt_s0}"
OFFLINE_STEPS="${OFFLINE_STEPS:-100000}"
ONLINE_STEPS="${ONLINE_STEPS:-50000}"
EVAL_STEPS="${EVAL_STEPS:-0,10000,25000,50000}"
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

mkdir -p "$LOGROOT"
cd "$REPO" || exit 1
SOURCE_COMMIT=$(git rev-parse HEAD)

ts() { TZ=Asia/Seoul date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(ts)" "$*" | tee -a "$LOGROOT/orchestrator.log"; }
tag_num() { printf '%s' "$1" | tr '.' 'p' | tr '-' 'm'; }

if [[ -f "$LOGROOT/STOP" ]]; then
  log "FATAL stale STOP marker: remove $LOGROOT/STOP only after checking no jobs are live"
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
# Sparse four-point search used by the paper, not the full Cartesian NT grid.
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

job_tag() {
  local phase="$1" env="$2" gap="$3" n="${4:-}" temp="${5:-}"
  if [[ "$phase" == anchor ]]; then
    printf 'pbf_anchor_%s_gap%s_s%s' "$env" "$gap" "$SEED" | tr '-' '_'
  else
    printf 'pbf_online_%s_gap%s_N%s_T%s_s%s' \
      "$env" "$gap" "$n" "$(tag_num "$temp")" "$SEED" | tr '-' '_'
  fi
}

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
    pid="${PIDS[$i]}"
    if ! kill -0 "$pid" 2>/dev/null; then
      key="${KEYS[$i]}"
      wait "$pid" 2>/dev/null && rc=0 || rc=$?
      log "DONE key=$key pid=$pid rc=$rc"
      if ((rc == 0)); then
        touch "$LOGROOT/$key.DONE"
      else
        printf 'rc=%s\n' "$rc" >"$LOGROOT/$key.FAIL"
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
  log "STOP requested; sending one clean TERM to each job process group"
  for pid in "${PIDS[@]:-}"; do
    kill -TERM -- "-$pid" 2>/dev/null || true
  done
}
trap 'touch "$LOGROOT/STOP"; clean_stop' INT TERM

find_anchor() {
  local env="$1" gap="$2" root checkpoint
  root="$SAVE_DIR/pixel_o2o/${RUN_GROUP}_anchor_gap${gap}"
  checkpoint=$(find "$root" -type f \
    -path "*/pixel_pathbridger_online_idm_${env}_sd$(printf '%03d' "$SEED")_*/checkpoints/step_0.pkl" \
    -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | cut -d' ' -f2-)
  [[ -n "$checkpoint" && -f "$checkpoint" ]] || return 1
  printf '%s\n' "$checkpoint"
}

launch_job() {
  local phase="$1" env="$2" gap="$3" n="${4:-1}" temp="${5:-0}"
  local key slot gpu cpu0 cpu1 out cfg group anchor pid
  local -a cmd
  if [[ "$(git rev-parse HEAD)" != "$SOURCE_COMMIT" ]] || ! git diff --quiet; then
    log "FATAL tracked source changed after queue start; refusing mixed-code sweep"
    return 1
  fi
  key=$(job_tag "$phase" "$env" "$gap" "$n" "$temp")
  if [[ -f "$LOGROOT/$key.DONE" ]]; then
    log "SKIP done $key"
    return 0
  fi
  if [[ -f "$LOGROOT/$key.LOCK" ]]; then
    pid=$(cat "$LOGROOT/$key.LOCK" 2>/dev/null)
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      log "SKIP live lock $key pid=$pid"
      return 0
    fi
    rm -f "$LOGROOT/$key.LOCK"
  fi
  while ((${#PIDS[@]} >= MAX)); do
    reap_one || sleep 5
  done
  slot=$(next_slot) || return 1
  gpu=${SLOT_GPUS[$slot]}
  cpu0=${CPU_BASES[$slot]}
  cpu1=$((cpu0 + CPU_STRIDE - 1))
  out="$LOGROOT/$key.out"

  if [[ "$phase" == anchor ]]; then
    group="${RUN_GROUP}_anchor_gap${gap}"
    cfg=$(printf '{"endpoint_value_scale":%s,"eval_num_candidates":1,"eval_temperature":0.0,"tune_tag":"anchor_gap%s"}' "$gap" "$gap")
    cmd=("$PY" -u train_pixel.py
      --algorithm=pixel_pathbridger_online_idm --env_name="$env" --seed="$SEED"
      --run_group="$group" --protocol_suite=pixel_pbf_gap_anchor
      --dataset_dir="$DATASET_DIR" --save_dir="$SAVE_DIR"
      --offline_steps="$OFFLINE_STEPS" --online_steps=0 --random_steps=0
      --eval_steps=0 --eval_episodes=0 --frame_stack=3 --nouse_tqdm
      --config_json="$cfg")
  else
    anchor=$(find_anchor "$env" "$gap") || {
      log "FATAL missing anchor env=$env gap=$gap"
      return 1
    }
    group="${RUN_GROUP}_online_gap${gap}_N${n}_T$(tag_num "$temp")"
    cfg=$(printf '{"endpoint_value_scale":%s,"eval_num_candidates":%s,"eval_temperature":%s,"tune_tag":"gap%s_N%s_T%s"}' "$gap" "$n" "$temp" "$gap" "$n" "$(tag_num "$temp")")
    cmd=("$PY" -u train_pixel.py
      --algorithm=pixel_pathbridger_online_idm --env_name="$env" --seed="$SEED"
      --run_group="$group" --protocol_suite=pixel_pbf_gap_nt_4search
      --dataset_dir="$DATASET_DIR" --save_dir="$SAVE_DIR"
      --restore_path="$anchor" --restore_step=0
      --online_steps="$ONLINE_STEPS" --random_steps=10000
      --replay_capacity=50000 --frame_stack=3 --her_probability=0.8
      --eval_steps="$EVAL_STEPS" --eval_episodes=10
      --resume_interval=50000 --resume_keep=2 --save_replay --nouse_tqdm
      --config_json="$cfg")
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

run_phase() {
  local phase="$1" env gap cell n temp
  log "PHASE_START $phase"
  for env in "${ENVS[@]}"; do
    for gap in "${GAPS[@]}"; do
      if [[ "$phase" == anchor ]]; then
        launch_job anchor "$env" "$gap" || return 1
      else
        for cell in "${NT_SEARCH[@]}"; do
          IFS='|' read -r n temp <<<"$cell"
          launch_job online "$env" "$gap" "$n" "$temp" || return 1
        done
      fi
      [[ -f "$LOGROOT/STOP" ]] && { clean_stop; break 2; }
    done
  done
  while ((${#PIDS[@]} > 0)); do
    [[ -f "$LOGROOT/STOP" ]] && clean_stop
    reap_one || sleep "$POLL_SEC"
  done
  [[ ! -f "$LOGROOT/STOP" ]]
}

printf '%s\n' $$ >"$LOGROOT/orchestrator.pid"
log "START commit=$SOURCE_COMMIT anchors=8 online_branches=32 GPUs=${GPU_LIST[*]} slots_per_gpu=$SLOTS_PER_GPU offline=$OFFLINE_STEPS online=$ONLINE_STEPS"
run_phase anchor && run_phase online
rc=$?
if ((rc == 0)); then log "QUEUE_COMPLETE"; else log "QUEUE_STOPPED_OR_FAILED rc=$rc"; fi
rm -f "$LOGROOT/orchestrator.pid"
log "EXIT"
exit "$rc"
