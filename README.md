# PathBridger

PathBridger is an offline goal-conditioned reinforcement-learning method that
plans a short state-space path and decodes its first transitions with inverse
dynamics. This repository contains the canonical state implementation and the
full-action visual comparison track for OGBench.

![PathBridger bridge-policy architecture](assets/architecture.jpg)

## What is in this repository?

There are two primary experiment tracks:

| Track | Entry point | Methods | Offline information |
|---|---|---|---|
| State PathBridger | `main.py`, `evaluate.py` | PBF, PBG, stochastic-prefix ablations | states, actions, episode boundaries |
| Full-action visual offline RL | `train_pixel.py`, `evaluate_pixel.py` | Pixel PBF, visual HIQL, visual OTA | RGB frames, actions, episode boundaries |

The action-free offline-to-online study lives in the separate
[Action-Free PathBridger repository](https://github.com/SChoish/Action_Free_PathBridger).
Shared action-free utilities remain here for checkpoint and benchmark
compatibility, but new action-free experiments should use that repository.

## Method overview

State PathBridger jointly learns four components:

1. a bounded transitive state-goal value and its EMA target;
2. a value-weighted, goal-conditioned endpoint distribution;
3. an endpoint-pinned residual bridge in state space; and
4. an inverse-dynamics model that decodes the first five bridge transitions.

The two endpoint variants share the same value, bridge, and IDM:

- **PBF** models endpoint displacement with conditional rectified flow.
- **PBG** models endpoint displacement with a conditional diagonal Gaussian.

The default bridge prefix is deterministic. `low_rank_gaussian` and
`joint_flow` provide joint stochastic residuals over the five executed bridge
states. These are prefix models, not full-horizon trajectory generators.

State PathBridger contains no neural actor, action-conditioned critic,
Triangle-Q module, or actor-finetuning phase.

## Installation

Python 3.10 or newer is required. Install the JAX build appropriate for your
CPU or accelerator, then install the project:

```bash
git clone https://github.com/SChoish/PathBridger_dist.git
cd PathBridger_dist
python -m venv .venv
source .venv/bin/activate
pip install -c constraints-tested.txt -e ".[dev]"
```

Weights & Biases support is optional:

```bash
pip install -e ".[tracking]"
```

OGBench data is not silently downloaded by the visual runners. Pass an
existing `--dataset_dir`, or explicitly opt in with
`--allow_dataset_download=true` where supported.

## Quick start: state PBF/PBG

Train the default PBF configuration:

```bash
python main.py --seed=0
```

Choose one of the eight environment-specific PBF or PBG configurations:

```bash
python main.py \
  --agent=configs/pbf/cube_double.py \
  --run_group=paper \
  --seed=0

python main.py \
  --agent=configs/pbg/puzzle_4x4.py \
  --run_group=paper \
  --seed=0
```

The paper configurations cover:

```text
antmaze-medium-navigate-v0   antmaze-large-navigate-v0
cube-single-play-v0          cube-double-play-v0
cube-triple-play-v0          puzzle-3x3-play-v0
puzzle-4x4-play-v0           scene-play-v0
```

Each config fixes the endpoint horizon, discount, endpoint-value scale,
distance weighting, Best-of-N candidate count, and sampling temperature for
that environment. The default run uses one million updates, batch size 1024,
and a 100k checkpoint interval.

Evaluate a checkpoint on the five official OGBench tasks:

```bash
python evaluate.py \
  --agent=configs/pbf/cube_double.py \
  --checkpoint_dir=exp/pathbridger/paper/<run>/checkpoints \
  --checkpoint_step=1000000 \
  --episodes=50
```

An exact `params_<step>.pkl` path may be passed instead of a directory. The
step is then inferred from the filename.

### Resume and soft-stop

`main.py` handles `SIGTERM`, `SIGINT`, and `SIGHUP` by completing the current
update and writing an emergency checkpoint. A unified checkpoint contains the
agent, optimizer, JAX key, and host-sampler RNG, so offline training can resume
without restarting its batch stream:

```bash
python main.py \
  --agent=configs/pbf/cube_double.py \
  --restore_path=exp/pathbridger/paper/<run>/checkpoints \
  --restore_step=900000 \
  --run_dir=exp/pathbridger/paper/<run>
```

Host sampling is prefetched on one worker by default. Disable it with
`--noasync_prefetch` only when debugging the input pipeline.

### Stochastic bridge-prefix ablations

```bash
# Low-rank joint Gaussian.
python main.py --agent=configs/pbf/cube_double.py \
  --agent.prefix_model=low_rank_gaussian

# Joint rectified flow.
python main.py --agent=configs/pbf/cube_double.py \
  --agent.prefix_model=joint_flow
```

Keep `path_weight_beta=0` for a clean deterministic/Gaussian/flow backend
comparison. Temporal-geodesic path weighting is an independent ablation and
should be reported separately.

## Full-action visual offline RL

The visual track uses one-frame RGB observations for the official offline
baselines and never enters an online interaction phase:

| Registry name | Core model | Offline actions | Online updates |
|---|---|---:|---:|
| `pixel_pbf` | IMPALA-small + compact latent PBF + IDM | yes | none |
| `pixel_hiql` | visual HIQL hierarchy and AWR actors | yes | none |
| `pixel_ota` | option-aware visual hierarchy | yes | none |

Train Pixel PBF:

```bash
python train_pixel.py \
  --algorithm=pixel_pbf \
  --env_name=visual-cube-double-play-v0 \
  --dataset_dir=/path/to/ogbench_visual \
  --offline_steps=500000 \
  --online_steps=0 \
  --frame_stack=1 \
  --seed=0
```

Train the two required hierarchical baselines:

```bash
bash scripts/run_pixel_hiql_ota.sh \
  pixel_hiql visual-cube-double-play-v0
bash scripts/run_pixel_hiql_ota.sh \
  pixel_ota visual-cube-double-play-v0
```

Pixel PBF uses a 512-D IMPALA visual feature and a 32-D length-normalized path
representation. TransV and IDM train the representation; endpoint and bridge
geometry are stop-gradient with respect to the encoder. Raw `uint8` frames
remain GPU-resident, and each update transfers compact indices and actions.

Collapse diagnostics—effective rank, cross-sample variance, temporal and
random-pair distances, IDM error, and action statistics—are evaluated on log
steps. Ordinary updates use the same loss and gradients without the log-only
eigendecomposition.

Run the production-shape GPU smoke test before launching a sweep:

```bash
PYTHONPATH=. python scripts/smoke_pixel_pbf_gpu.py
```

The smoke reports fast-step and diagnostic-step throughput separately and
fails on non-finite actions or representation diagnostics. See
[`docs/pixel_pbf.md`](docs/pixel_pbf.md) for the architecture locks, checkpoint
compatibility rule, and promotion gate.

## Outputs and provenance

State runs are written below:

```text
exp/pathbridger/<run_group>/<experiment>/
├── flags.json
├── train.csv
└── checkpoints/params_<step>.pkl
```

Visual runs use `exp/pixel_o2o/<run_group>/<experiment>/` and include
`metadata.json`, offline/online CSV files as applicable, evaluation curves,
and phase-specific checkpoints. Metadata records the observation regime,
offline fields seen, trainable modules, frozen modules, protocol identifier,
and configuration hash.

Legacy raw-512 Pixel PBF checkpoints remain evaluation-only. They cannot be
resumed into the compact 32-D architecture; start a fresh training run instead.

## Repository layout

```text
agents/                  state and visual agents
configs/pbf/             eight state PBF configurations
configs/pbg/             eight state PBG configurations
docs/                    visual and action-free protocol specifications
envs/                    OGBench loading and modality checks
scripts/                 manifests, queues, smoke tests, and evaluators
tests/                   sampler, loss, checkpoint, and protocol tests
main.py                  state offline training
evaluate.py              state checkpoint evaluation
train_pixel.py           unified visual runner
evaluate_pixel.py        visual checkpoint evaluation
```

All state PathBridger losses and inference live in
[`agents/pathbridger.py`](agents/pathbridger.py). Visual method names,
information boundaries, and module-update contracts are centralized in
[`agents/pixel_registry.py`](agents/pixel_registry.py).

## Reproducibility checks

Run the complete test suite on CPU:

```bash
JAX_PLATFORMS=cpu pytest -q tests
```

Useful focused checks:

```bash
pytest -q tests/test_agent_smoke.py tests/test_datasets.py
pytest -q tests/test_pixel_algorithms.py tests/test_pixel_lapo.py
bash -n scripts/*.sh
```

The tests cover episode-safe sampling, state/PBF batch contracts, stochastic
prefix endpoint pinning, checkpoint round trips, visual information boundaries,
frozen-module audits, and fast/full-metric update parity.

## Documentation

- [`docs/pixel_pbf.md`](docs/pixel_pbf.md): Pixel PBF architecture and
  diagnostic gate.
- [`docs/pixel_baselines.md`](docs/pixel_baselines.md): visual HIQL, OTA, and
  controlled pixel baseline notes.
- [`docs/pixel_benchmark_protocol.md`](docs/pixel_benchmark_protocol.md):
  action-free pixel benchmark protocol shared with the companion repository.
- [`docs/action_free_o2o_protocol.md`](docs/action_free_o2o_protocol.md): state
  action-free offline-to-online protocol.
- [`third_party_refs.json`](third_party_refs.json): audited baseline provenance.

## License

See [`LICENSE`](LICENSE).
