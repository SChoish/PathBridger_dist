# Action-free offline-to-online protocol

## Information contract

One goal-conditioned policy is trained per environment.  The strict
action-free methods may read only compact state trajectories and episode
boundaries during offline pretraining:

```text
allowed:    observations, terminals
forbidden:  actions, logged rewards, returns, behavior policy
```

Sparse goal rewards used for value learning are reconstructed from sampled
state-goal relations using one transition convention: reward is zero and the
Bellman mask is terminal when the achieved next state reaches the goal.
Current-state goals are not sampled as failures; immediate future goals supply
positive anchors.  The immutable state buffer and the action-bearing
online replay are separate objects.  Only the explicitly labeled
`gc_sac_50_50`
upper bound receives the offline action array.

For `pbf_online_idm`, PBF is constructed without an IDM module.  Its value,
endpoint-flow, and bridge parameter tree is frozen for the entire online
phase.  A new deterministic IDM is initialized at online step zero; it is the
only optimized module.  A SHA-256 parameter-tree audit fails the run if the
PBF tree changes.

The default `--pbf_execute_horizon=1` replans every primitive step.
`--pbf_execute_horizon=5` uses a real collection-time action queue: all five
decoded primitive transitions are executed and stored before replanning.  This
is deterministic PBF prefix execution, not a stochastic PathFlower claim.  The
five actions are decoded from the planned state pairs before execution, so this
setting is explicitly **open-loop**; feedback decoding is reserved for a later
ablation and is not conflated with the main method.

## Main methods and provenance

| Result block | Name | Implementation status | Native online update |
|---|---|---|---|
| Proposed | `pbf_online_idm` | Native PBF extension | IDM only |
| Action-free | `gc_mscp_style` | Simplified paper reimplementation; official source pinned | low policy/value, 50:50 state mixing |
| Action-free | `hiql_endpoint_online` | HIQL-inspired endpoint hierarchy | low policy/value |
| Action-free | `gc_af_guide` | Goal-conditioned adaptation | Guided SAC and guide critic |
| Action-free | `gc_oso_decqn_factorized` | Factorized three-bin reimplementation and GC adaptation | TD3, IDM, guide switching |
| Online only | `gc_sac` | Shared native implementation | SAC + future HER |
| Online only | `gc_td3` | Shared native implementation | TD3 + future HER |
| Full action | `gc_sac_50_50` | Symmetric-replay upper bound; not RLPD | SAC on 50:50 offline/online replay |

The adaptation and reimplementation labels are intentional: they are not
presented as exact reproduction numbers from a different benchmark.  Audited
repositories and commits are stored in `third_party_refs.json` and emitted in
run metadata where applicable.

## Offline training

Method-native default update counts are used:

| Method | Offline updates |
|---|---:|
| PBF-OnlineIDM | 1,000,000 |
| GC-MSCP | 1,000,000 |
| Passive-HIQL | 1,000,000 |
| GC-AF-Guide | 50,000 |
| GC-OSO-DecQN | 3,000,000 |
| GC-SAC / GC-TD3 | 0 |

`--offline_steps` can override these defaults for smoke tests and equal-budget
appendices.  Online results must not be used to select an offline checkpoint.

## Online training

- Budget: 1,000,000 primitive environment steps.
- Initial grounding: 10,000 random-action steps, counted in the budget.
- First update: replay size 1,000 by default.
- Replay capacity: 1,000,000 real online transitions.
- Goal distribution: uniform over OGBench task IDs 1--5 at episode reset.
- Goal policy: a single `pi(a | s, g)` per environment.
- Replay relabeling: episode-aware future HER for action-policy baselines.
- HER includes the current transition's achieved next state as a candidate and
  logs relabel, HER-success, and commanded-success fractions.
- AF-Guide stores an explicit desired-state validity bit; random-grounding
  samples do not update its guide critic or guide the actor.
- Evaluation trajectories are never inserted into replay.
- Task-goal RNG and exploration RNG are independent, preserving paired goal
  sequences across algorithms with the same seed.

The proposed online IDM uses a three-layer 512-unit MLP, L1 loss, batch size
512, and learning rate `1e-3`.  Its targets contain only real online
`(s, a, s')` transitions.

## Benchmark and evaluation

The locked main suite contains eight state-based OGBench environments:

```text
antmaze-medium-navigate-v0   antmaze-large-navigate-v0
cube-single-play-v0          cube-double-play-v0
cube-triple-play-v0          puzzle-3x3-play-v0
puzzle-4x4-play-v0           scene-play-v0
```

Run five seeds.  Evaluate at online steps
`0, 10k, 25k, 50k, 100k, 250k, 500k, 1M`.  At each checkpoint, run ten
episodes for each official task ID (50 episodes per environment and seed),
clip actions to the environment bounds, and report any-step episode success.

Primary reporting is IQM AUC@250k with a stratified-bootstrap 95% confidence
interval.  Also report AUC@1M and Success@1M.  The aggregator refuses to
extrapolate interrupted runs to the requested final budget and keeps
online-only, action-free, and full-action result blocks separate.  Its
bootstrap preserves the `[seed, environment]` matrix, resamples the seed axis
within each environment, and also reports pairwise probability of improvement.

## Locked runs

Start with the required 21-run P0 gate (7 methods x 3 environments x 1 seed),
including Scene to exercise its non-indexable oracle goal projection:

```bash
python scripts/make_af_manifest.py --suite=p0_smoke --output=p0_smoke.csv
```

It uses 50k online steps and evaluates at 0/10k/25k/50k.  Inspect HER positive
rate, IDM error, desired-next realization error, AF-Guide valid-target rate,
value targets, critic Q statistics, successes, and the planner freeze audit.
Only then generate the 84-run four-environment, three-seed pilot:

```bash
python scripts/make_af_manifest.py --suite=pilot --output=pilot.csv
```

The 280-run eight-environment, five-seed suite is a **pre-tuning screening
track**, not a final fair leaderboard:

```bash
python scripts/make_af_manifest.py \
  --suite=screening \
  --output=screening.csv
```

PBF currently uses paper-selected environment-specific hyperparameters while
the baseline ports use global defaults.  Final native/tuned claims therefore
require an equal per-method tuning budget and validation-only selection.  This
repository records that limitation instead of treating screening results as a
finished comparison.

Add the full-action upper-bound runs only when requested:

```bash
python scripts/make_af_manifest.py \
  --suite=screening \
  --include_full_action \
  --output=benchmark_with_upper_bound.csv
```

Every run writes immutable provenance metadata, offline and online training
logs, evaluation curves, and component-wise checkpoints.  Aggregate completed
runs with `python aggregate_results.py --root=<experiment-root>`.
