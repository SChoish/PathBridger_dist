#!/usr/bin/env python3
"""GPU Best-(N,T) eval with optional --prefix-samples (M sweep).

Same artifacts as ``cpu_eval_pbf_pathwt_ckpt.py`` so CPU/GPU workers share
DONE/LOCK markers. Sets CUDA before importing the CPU entrypoint (which only
``setdefault``s device env vars).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--gpu', type=str, default=os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
_pre_args, _ = _pre.parse_known_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(_pre_args.gpu)
# Must set before importing cpu_eval (it setdefaults JAX_PLATFORMS=cpu).
os.environ['JAX_PLATFORMS'] = 'cuda'
os.environ.setdefault('MUJOCO_GL', os.environ.get('MUJOCO_GL', 'egl'))

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_cpu_path = _REPO / 'scripts' / 'cpu_eval_pbf_pathwt_ckpt.py'
_spec = importlib.util.spec_from_file_location('cpu_eval_pbf_pathwt_ckpt', _cpu_path)
_cpu = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_cpu)


def main() -> None:
    cleaned: list[str] = []
    skip = False
    for a in sys.argv:
        if skip:
            skip = False
            continue
        if a == '--gpu':
            skip = True
            continue
        if a.startswith('--gpu='):
            continue
        cleaned.append(a)
    sys.argv = cleaned
    _cpu.main()


if __name__ == '__main__':
    main()
