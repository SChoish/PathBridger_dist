"""Evaluation-loop tests using a tiny fake environment and agent."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


jax = pytest.importorskip("jax")

from utils.evaluation import evaluate  # noqa: E402


class _FakeAgent:
    def __init__(self):
        self.calls = []

    def sample_action_chunks(
        self,
        *,
        observations,
        goals,
        num_candidates,
        temperature,
        seed,
    ):
        self.calls.append(
            {
                "observations": np.asarray(observations),
                "goals": np.asarray(goals),
                "num_candidates": num_candidates,
                "temperature": temperature,
                "seed": seed,
            }
        )
        batch_size = np.asarray(observations).shape[0]
        # Exercise clipping on both action dimensions.
        return np.broadcast_to(
            np.asarray([3.0, -3.0], dtype=np.float32),
            (batch_size, 5, 2),
        )


class _FakeEnv:
    def __init__(self):
        self.action_space = SimpleNamespace(
            low=np.asarray([-1.0, -0.5], dtype=np.float32),
            high=np.asarray([1.0, 0.5], dtype=np.float32),
        )
        self.spec = SimpleNamespace(max_episode_steps=7)
        self.actions = []
        self.reset_options = []
        self.episode_step = 0

    def reset(self, *, options):
        self.reset_options.append(dict(options))
        self.episode_step = 0
        task_id = float(options["task_id"])
        observation = np.asarray([task_id, 0.0, 0.0], dtype=np.float32)
        goal = np.asarray([task_id, 1.0, 1.0], dtype=np.float32)
        return observation, {"goal": goal}

    def step(self, action):
        self.actions.append(np.asarray(action))
        self.episode_step += 1
        success = self.episode_step == 2
        terminated = self.episode_step >= 3
        observation = np.asarray(
            [0.0, float(self.episode_step), 0.0],
            dtype=np.float32,
        )
        return observation, 0.0, terminated, False, {"success": success}


def test_evaluate_runs_standard_task_resets_chunks_clipping_and_success():
    agent = _FakeAgent()
    env = _FakeEnv()

    metrics = evaluate(
        agent,
        env,
        task_ids=(2, 4),
        episodes_per_task=2,
        num_candidates=7,
        temperature=0.25,
        seed=11,
    )

    assert metrics == {
        "task_2_success": 1.0,
        "task_4_success": 1.0,
        "overall_success": 1.0,
        "num_tasks": 2,
        "episodes_per_task": 2,
    }
    assert env.reset_options == [
        {"task_id": 2, "render_goal": False},
        {"task_id": 2, "render_goal": False},
        {"task_id": 4, "render_goal": False},
        {"task_id": 4, "render_goal": False},
    ]
    assert len(agent.calls) == 4
    assert all(call["observations"].shape == (1, 3) for call in agent.calls)
    assert all(call["goals"].shape == (1, 3) for call in agent.calls)
    assert all(call["num_candidates"] == 7 for call in agent.calls)
    assert all(call["temperature"] == 0.25 for call in agent.calls)
    np.testing.assert_allclose(
        np.stack(env.actions),
        np.broadcast_to([1.0, -0.5], (12, 2)),
    )
