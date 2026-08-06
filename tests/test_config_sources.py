"""Dependency-free source invariants for paper configs and agent scope."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# (K, gamma, gap scale, distance exponent, candidates, temperature)
PAPER_BEST_CONFIGS = {
    ("pbf", "antmaze_medium"): (25, 0.99, 10.0, 0.0, 2, 0.25),
    ("pbf", "antmaze_large"): (25, 0.995, 10.0, 0.0, 16, 0.5),
    ("pbf", "cube_single"): (40, 0.99, 5.0, 0.7, 1, 0.0),
    ("pbf", "cube_double"): (40, 0.99, 10.0, 1.0, 2, 0.25),
    ("pbf", "cube_triple"): (40, 0.995, 10.0, 1.0, 1, 0.0),
    ("pbf", "puzzle_3x3"): (25, 0.99, 10.0, 0.5, 32, 1.0),
    ("pbf", "puzzle_4x4"): (25, 0.99, 10.0, 2.0, 32, 1.0),
    ("pbf", "scene"): (25, 0.99, 5.0, 1.0, 16, 0.5),
    ("pbg", "antmaze_medium"): (25, 0.99, 10.0, 0.0, 1, 0.0),
    ("pbg", "antmaze_large"): (25, 0.995, 10.0, 0.0, 1, 0.0),
    ("pbg", "cube_single"): (25, 0.99, 10.0, 0.7, 1, 0.0),
    ("pbg", "cube_double"): (25, 0.99, 10.0, 1.0, 1, 0.0),
    ("pbg", "cube_triple"): (25, 0.995, 10.0, 1.0, 1, 0.0),
    ("pbg", "puzzle_3x3"): (40, 0.99, 10.0, 0.5, 2, 0.25),
    ("pbg", "puzzle_4x4"): (40, 0.995, 10.0, 2.0, 16, 0.5),
    ("pbg", "scene"): (40, 0.99, 5.0, 1.0, 16, 0.5),
}


def _best_config_keywords(path: Path) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    get_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_config"
    )
    returned = next(
        node.value for node in get_config.body if isinstance(node, ast.Return)
    )
    assert isinstance(returned, ast.Call)
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in returned.keywords
        if keyword.arg is not None
    }


def _literal_assignment(tree: ast.AST, name: str):
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} was not assigned")


def _agent_default_config() -> dict[str, object]:
    path = PROJECT_ROOT / "agents" / "pathbridger.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    get_config = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_config"
    )
    returned = next(
        node.value for node in get_config.body if isinstance(node, ast.Return)
    )
    assert isinstance(returned, ast.Call)
    assert len(returned.args) == 1
    payload = returned.args[0]
    assert isinstance(payload, ast.Call)
    assert isinstance(payload.func, ast.Name) and payload.func.id == "dict"
    return {
        keyword.arg: ast.literal_eval(keyword.value)
        for keyword in payload.keywords
        if keyword.arg is not None
    }


def test_all_sixteen_configs_match_the_paper_best_parameter_table():
    discovered = {
        (family, path.stem)
        for family in ("pbf", "pbg")
        for path in (PROJECT_ROOT / "configs" / family).glob("*.py")
        if path.name != "__init__.py"
    }
    assert discovered == set(PAPER_BEST_CONFIGS)

    distribution = {"pbf": "flow", "pbg": "gaussian"}
    fields = (
        "horizon",
        "discount",
        "endpoint_value_scale",
        "value_distance_weight_power",
        "eval_num_candidates",
        "eval_temperature",
    )
    for (family, task), expected in PAPER_BEST_CONFIGS.items():
        values = _best_config_keywords(
            PROJECT_ROOT / "configs" / family / f"{task}.py"
        )
        assert values["endpoint_distribution"] == distribution[family]
        assert tuple(values[field] for field in fields) == expected


def test_shared_goal_mixes_use_paper_four_tuple_order_without_geom_flags():
    defaults = _agent_default_config()
    assert defaults["actor_p"] == (0.0, 0.0, 1.0, 0.0)
    assert defaults["critic_p"] == (0.0, 1.0, 0.0, 0.0)
    assert defaults["path_weight_beta"] == 0.0
    assert defaults["path_weight_min"] == 0.1
    assert defaults["path_weight_warmup"] == 100_000
    assert defaults["path_weight_ramp"] == 100_000
    assert defaults["path_distance_cap_multiplier"] == 2.0
    assert defaults["prefix_model"] == "deterministic"
    assert defaults["prefix_loss_weight"] == 1.0
    assert defaults["prefix_rank"] == 8
    assert defaults["prefix_sigma_floor"] == 1e-3
    assert defaults["prefix_scale_floor"] == 1e-3
    assert defaults["prefix_flow_steps"] == 8
    assert defaults["eval_prefix_selection"] == "sample_one"
    assert defaults["eval_num_prefix_samples"] == 1
    assert defaults["eval_prefix_temperature"] == 1.0
    assert defaults["eval_include_deterministic_prefix"] is False

    source_paths = (
        PROJECT_ROOT / "agents" / "pathbridger.py",
        PROJECT_ROOT / "utils" / "datasets.py",
        PROJECT_ROOT / "configs" / "_base.py",
    )
    source = "\n".join(path.read_text(encoding="utf-8") for path in source_paths)
    assert "actor_geom_sample" not in source
    assert "critic_geom_sample" not in source
    assert "value_geom_sample" not in source


def test_sampler_and_agent_share_the_exact_thirteen_key_contract():
    agent_path = PROJECT_ROOT / "agents" / "pathbridger.py"
    tree = ast.parse(agent_path.read_text(encoding="utf-8"), filename=str(agent_path))
    required = _literal_assignment(tree, "_REQUIRED_BATCH_KEYS")
    assert required == (
        "observations",
        "next_observations",
        "actions",
        "bridge_targets",
        "endpoint_goals",
        "endpoint_targets",
        "value_goals",
        "value_offsets",
        "base_goals",
        "base_offsets",
        "transitive_subgoals",
        "transitive_offsets",
        "transitive_valids",
    )

    dataset_path = PROJECT_ROOT / "utils" / "datasets.py"
    dataset_tree = ast.parse(
        dataset_path.read_text(encoding="utf-8"),
        filename=str(dataset_path),
    )
    sample_method = next(
        node
        for node in ast.walk(dataset_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "sample"
        and any(
            isinstance(parent, ast.ClassDef) and parent.name == "PathBridgerDataset"
            for parent in dataset_tree.body
            if node in getattr(parent, "body", ())
        )
    )
    returned = next(
        node.value
        for node in ast.walk(sample_method)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    )
    returned_keys = tuple(ast.literal_eval(key) for key in returned.keys)
    assert returned_keys == required


def test_agent_source_contains_no_neural_actor_or_action_q_module():
    path = PROJECT_ROOT / "agents" / "pathbridger.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    class_names = {node.name.lower() for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert not any("actor" in name or "critic" in name or "actionq" in name for name in class_names)

    identifiers = {
        node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    banned_identifiers = {
        "actor",
        "target_actor",
        "action_q",
        "action_value",
        "critic",
    }
    assert banned_identifiers.isdisjoint(identifiers | attributes)

    required = set(_literal_assignment(tree, "_REQUIRED_BATCH_KEYS"))
    assert {
        "actor_goals",
        "actor_actions",
        "action_value_goals",
        "action_chunks",
    }.isdisjoint(required)
