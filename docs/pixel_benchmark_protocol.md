# Action-free pixel offline-to-online benchmark

This track is isolated from the state-vector `af_o2o_v3` benchmark. Pixel
results use protocol family `pixel_o2o_v3`, a separate output root, manifests,
checkpoints, and aggregate tables.

## Implemented comparison set

| Registry name | Offline input | Offline training | Online updates | Status |
| --- | --- | --- | --- | --- |
| `gc_pixel_drqv2` | none | none | encoder, actor, critic | goal-image OGBench adaptation of DrQ-v2 |
| `vip_style_frozen_gc_drqv2` | RGB, terminals | VIP-style temporal value encoder | actor, critic | controlled adaptation; encoder frozen online |
| `vip_style_finetuned_gc_drqv2` | RGB, terminals | VIP-style temporal value encoder | encoder, actor, critic | controlled adaptation; encoder fine-tuned online |
| `gc_pixel_lapo_decoder` | RGB, terminals | VQ latent action model and goal-conditioned latent policy | action decoder | controlled LAPO-Decoder adaptation, not native online PPO |
| `gc_pixel_apv_style_drq` | RGB, terminals | next-latent video prediction and pixel reconstruction | encoder, actor, critic, action-conditioned dynamics | APV-style + GC-DrQ, not native DreamerV2 APV |

Online learning is not globally restricted to an IDM. Each method may update
the modules required by its declared algorithm. The exact module list is part
of registry metadata and every checkpoint. The controlled LAPO variant still
updates only its action decoder online because that is the intended grounding
comparison, while VIP-finetuned, DrQ-v2, and APV-style update broader online
models.

These implementations preserve the original methods' comparison axes but are
not claimed as official reproductions. In particular, VIP is pretrained on the
selected OGBench videos rather than using the released Ego4D representation,
and APV-style uses the shared JAX goal-conditioned DrQ controller rather than
the official TensorFlow DreamerV2 stack. Native ports, if added, must receive
different registry names and a separate result block.

The proposed entry learns a five-state latent path from action-free frame
sequences. Its encoder, target encoder, bridge, and world decoder are frozen
for the complete online phase. Only a separately initialized inverse dynamics
model is grounded with newly executed RGB/action/RGB transitions. The registry
and checkpoint metadata expose this module boundary, and training hash-checks
the frozen modules.

## Information boundary

For every strict action-free method, the offline loader constructs an immutable
view containing exactly `observations: uint8[N,H,W,3]` and `terminals`. It
discards logged actions, rewards, simulator states, and privileged metadata
before agent construction. Future goals, sparse temporal rewards, and masks are
derived solely from episode order. `gc_pixel_drqv2` does not open an offline
dataset at all.

New actions and environment success enter only after online interaction begins.
The online replay stores each raw RGB frame once and transitions refer to frame
IDs. It also records episode ID, timestep, behavior-goal frame ID, executed
action, sparse reward, and mask. Sampling constructs three-frame histories and
applies episode-local future-image HER without crossing reset boundaries.
Behavior goals, rewards, and masks remain separately available for methods that
need the commanded transition. This is an interaction-grounded regime and must
not be pooled with offline-action or mixed-data upper bounds.

## Locked online protocol

- Environments: the eight official `visual-*` OGBench counterparts listed in
  `scripts/make_pixel_manifest.py`.
- Goal distribution: uniform over official task IDs 1--5 at episode reset.
- Evaluation: task IDs 1--5, ten episodes per task, deterministic policy,
  any-step success.
- Online budget: primitive environment steps, shared by all methods.
- Default checkpoints: 0, 10k, 25k, 50k, 100k, 250k, 500k, and 1M steps where
  the suite budget reaches them.
- Exploration bootstrap: 10k shared uniform-random steps, followed by each
  declared policy's action noise.
- Observation history: three consecutive RGB frames, channel-stacked; history
  is left-padded with the episode's first available frame.
- Replay: 50k frame-indexed transitions with shared raw `uint8` frames.
- Goal relabeling: episode-aware future-image HER with probability 0.8. Replay
  logs relabel, relabeled-success, and commanded-success fractions.

The default offline update counts are algorithm-specific and are recorded in
metadata rather than counted against online environment steps. Offline compute
should additionally be reported as updates and wall-clock time in final tables.

## Suites and run counts

The unified manifest includes all six registry entries:

- `pilot`: 6 algorithms x 4 environments x 3 seeds = 72 runs.
- `screening`: 6 algorithms x 8 environments x 5 seeds = 240 runs.

Manifest generation never launches training or downloads datasets:

```bash
python scripts/make_pixel_manifest.py --suite=pilot \
  --output=manifests/pixel_pilot.csv
```

The visual dataset must be provisioned explicitly. Missing data raises an
error unless `--allow_dataset_download=true` is supplied. The online-only
DrQ-v2 entry can run without opening a visual offline dataset.

Evaluate a saved checkpoint with:

```bash
python evaluate_pixel.py --checkpoint=/exact/path/to/step_10.pkl
```

Formal runs require finite losses, non-degenerate actions, positive HER success
rate, nonzero critic/IDM learning signals as applicable, and no frozen-module
hash violation. Commanded success is reported separately so HER positives
cannot be mistaken for environment success.

## Deferred methods

PVDR, LAOM, native DreamerV3/APV, and interaction-grounded BCO remain explicit
future ports. They are not aliases for the current shared-backbone adaptations
and must not appear in result manifests until their algorithm-specific losses,
online update rules, provenance, and tests are implemented. LAOM is most useful
once a distractor track exists; PVDR and native APV/Dreamer require materially
larger world-model ports.
