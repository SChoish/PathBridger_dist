# Pixel PBF protocol

`pixel_pbf` is the full-action RGB counterpart of state PathBridger. Offline
training reads only RGB observations, episode terminals, and recorded actions;
it never uses privileged simulator state.

## Collapse-resistant representation

The visual tower is IMPALA-small with a 512-D feature layer. A learned linear
head maps this feature to a 32-D path representation, which is normalized to
radius `sqrt(32)`. TransV and the adjacent-transition IDM update the visual
tower and path head. Endpoint-flow and bridge losses receive stop-gradient
representations, so their regression targets cannot collapse the encoder.

Geometry targets are produced by the current online representation and then
stopped. The EMA encoder is used only with the EMA TransV target for bootstrap
values. Sampled flow endpoints and bridge states are projected back to the same
normalized manifold before value scoring or IDM decoding. Endpoint, bridge, and
IDM regression losses average over the feature/action axis so their scale does
not grow with representation dimension.

## Locked production recipe

- IMPALA feature dimension: 512
- path representation dimension: 32
- frame stack: 1
- batch size: 256
- training updates: 500k
- offline eval: 10 episodes/task at 250k, 25 episodes/task at 500k (final);
  post-hoc NT search also uses 25 episodes/task on the final checkpoint
- manipulation random crop: probability 0.5, one crop offset shared by every
  image and every path step in a sampled row
- path/action horizon: 5
- endpoint horizon: 40 for Cube and 25 for AntMaze, Puzzle, and Scene
- endpoint flow integration: 8 Euler steps
- learning rates: `3e-4`

The complete raw `uint8` frame store is copied to the GPU once. Each update
transfers only sampled indices, scalar offsets/masks, and actions; frame
stacking and crop augmentation run inside the compiled update.

## Required diagnostic gate

Training logs representation standard deviation, effective rank, one-step and
random-pair distances, IDM MSE and mean-action baseline MSE, and predicted
action magnitude/standard deviation. A run must not be promoted when the
representation distances or effective rank collapse, the endpoint value gap is
identically zero, or IDM MSE stays at the mean-action baseline.

These diagnostics are compiled into the `full_metrics=True` log-step variant,
not every training update. Regular updates keep the same loss and gradients
while skipping log-only reductions and the representation eigendecomposition.
`scripts/smoke_pixel_pbf_gpu.py` times both variants and checks that all gate
metrics remain finite.

Legacy raw-512 checkpoints remain loadable by `evaluate_pixel.py`, but cannot
be resumed into this architecture. They are scientifically separate runs.
