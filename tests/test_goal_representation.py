"""Goal-representation tests that also run when JAX is not installed."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "utils" / "goal_representation.py"
)


@pytest.fixture
def goal_module(monkeypatch):
    """Load the helpers with NumPy standing in for jax.numpy when necessary."""

    try:
        has_jax = importlib.util.find_spec("jax") is not None
    except (ImportError, ValueError):
        has_jax = False
    if not has_jax:
        fake_jax = types.ModuleType("jax")
        fake_jax.__path__ = []
        fake_jax.numpy = np
        monkeypatch.setitem(sys.modules, "jax", fake_jax)
        monkeypatch.setitem(sys.modules, "jax.numpy", np)

    spec = importlib.util.spec_from_file_location(
        "_pathbridger_goal_representation_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inferred_phi_indices_match_compact_ogbench_layouts(goal_module):
    assert goal_module.infer_phi_goal_obs_indices("antmaze-medium-navigate-v0", 29) == (
        0,
        1,
    )
    # Manipulation head=19, puzzle stride=4, cube stride=9.
    assert goal_module.infer_phi_goal_obs_indices("puzzle-3x3-play-v0", 31) == (
        20,
        24,
        28,
    )
    assert goal_module.infer_phi_goal_obs_indices("cube-double-play-v0", 37) == (
        19,
        20,
        21,
        28,
        29,
        30,
    )
    # Scene phi requires one-hot argmax operations and is not a plain slice.
    assert goal_module.infer_phi_goal_obs_indices("scene-play-v0", 40) == ()


def test_goal_representation_extracts_maze_puzzle_and_cube_phi(goal_module):
    maze = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.testing.assert_array_equal(
        np.asarray(
            goal_module.goal_representation(
                maze,
                "phi",
                env_name="antmaze-medium-navigate-v0",
            )
        ),
        maze[:, :2],
    )

    puzzle = np.arange(62, dtype=np.float32).reshape(2, 31)
    np.testing.assert_array_equal(
        np.asarray(
            goal_module.goal_representation(
                puzzle,
                "phi",
                env_name="puzzle-3x3-play-v0",
            )
        ),
        puzzle[:, [20, 24, 28]],
    )

    cube = np.arange(74, dtype=np.float32).reshape(2, 37)
    np.testing.assert_array_equal(
        np.asarray(
            goal_module.goal_representation(
                cube,
                "phi",
                env_name="cube-double-play-v0",
            )
        ),
        cube[:, [19, 20, 21, 28, 29, 30]],
    )


def test_scene_phi_matches_oracle_field_order(goal_module):
    # OGBench Scene: head 19, one cube (9), two 2-state buttons (4 each),
    # then drawer/window tail (4), for a total observation dimension of 40.
    goals = np.zeros((2, 40), dtype=np.float32)
    goals[:, 19:22] = np.asarray([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    goals[0, 28:30] = [0, 1]
    goals[0, 32:34] = [1, 0]
    goals[1, 28:30] = [1, 0]
    goals[1, 32:34] = [0, 1]
    goals[:, 36] = [0.25, 0.5]
    goals[:, 38] = [0.75, 1.0]

    phi = np.asarray(
        goal_module.goal_representation(
            goals,
            "phi",
            env_name="scene-play-v0",
        )
    )
    expected = np.asarray(
        [
            [1, 2, 3, 1, 0, 0.25, 0.75],
            [4, 5, 6, 0, 1, 0.5, 1.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(phi, expected)


def test_phi_validation_rejects_bad_indices_and_unknown_modes(goal_module):
    with pytest.raises(ValueError, match="non-negative"):
        goal_module.normalize_phi_goal_obs_indices((0, -1))
    with pytest.raises(TypeError, match="list or tuple"):
        goal_module.normalize_phi_goal_obs_indices("0,1")
    with pytest.raises(ValueError, match="unknown goal representation"):
        goal_module.assert_phi_goal_obs_indices(
            10,
            "legacy-mode",
            env_name="antmaze-medium-navigate-v0",
        )
