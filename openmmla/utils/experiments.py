from __future__ import annotations

import os

import yaml


def _find_project_root() -> str:
    """walk up from this file to find the repo root containing pyproject.toml."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isfile(os.path.join(d, "pyproject.toml")):
            return d
        d = os.path.dirname(d)
    return os.getcwd()


# ── experiments ──────────────────────────────────────────────────


def _experiments_path(project_root: str | None = None) -> str:
    root = project_root or _find_project_root()
    return os.path.join(root, "config", "experiments.yaml")


def load_experiments(project_root: str | None = None) -> dict:
    """load config/experiments.yaml from the project root.

    Returns the raw parsed dict, or an empty dict if the file is missing.
    """
    path = _experiments_path(project_root)
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def save_experiments(data: dict, project_root: str | None = None) -> None:
    """write *data* back to config/experiments.yaml."""
    path = _experiments_path(project_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def get_active_experiments(data: dict | None = None) -> list[dict]:
    """return the list of experiments whose status is 'active'."""
    if data is None:
        data = load_experiments()
    return [
        exp for exp in data.get("active_experiments", [])
        if exp.get("status") == "active"
    ]


def get_groups_for_experiment(exp_id: str, data: dict | None = None) -> list[str]:
    """derive distinct group_ids assigned to *exp_id*."""
    if data is None:
        data = load_experiments()
    assignments = data.get("assignments", {}).get(exp_id, {})
    groups: set[str] = set()
    for person_info in assignments.values():
        gid = person_info.get("group_id")
        if gid:
            groups.add(gid)
    return sorted(groups)


# ── tasks ────────────────────────────────────────────────────────


def _tasks_dir(project_root: str | None = None) -> str:
    root = project_root or _find_project_root()
    return os.path.join(root, "config", "tasks")


def list_tasks(project_root: str | None = None) -> list[str]:
    """return sorted list of task names (filename stems) in config/tasks/."""
    d = _tasks_dir(project_root)
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(d)
        if f.endswith((".yaml", ".yml")) and not f.startswith(".")
    )


def load_task(name: str, project_root: str | None = None) -> dict:
    """load a single task YAML by stem name."""
    d = _tasks_dir(project_root)
    for ext in (".yaml", ".yml"):
        path = os.path.join(d, name + ext)
        if os.path.isfile(path):
            with open(path, "r") as f:
                return yaml.safe_load(f) or {}
    return {}


def save_task(name: str, data: dict, project_root: str | None = None) -> None:
    """write task data to config/tasks/<name>.yaml."""
    d = _tasks_dir(project_root)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name + ".yaml")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def delete_task(name: str, project_root: str | None = None) -> bool:
    """delete config/tasks/<name>.yaml. returns True if file existed."""
    d = _tasks_dir(project_root)
    for ext in (".yaml", ".yml"):
        path = os.path.join(d, name + ext)
        if os.path.isfile(path):
            os.remove(path)
            return True
    return False


def select_experiment_and_group(project_root: str | None = None) -> tuple[str, str]:
    """interactive CLI prompts to pick an experiment and a group.

    Returns (exp_id, group_id).  Raises KeyboardInterrupt on cancel.
    """
    from .input import interactive_menu

    data = load_experiments(project_root)
    experiments = get_active_experiments(data)

    if not experiments:
        raise RuntimeError(
            "No active experiments found in config/experiments.yaml"
        )

    # --- select experiment ---
    exp_options = [e["experiment_id"] for e in experiments]
    exp_descriptions = [
        f"{e.get('title', '')} ({e.get('task_type', '')})"
        for e in experiments
    ]
    exp_idx = interactive_menu(
        "Select Experiment", exp_options, exp_descriptions, prompt_enter=False,
    )
    exp_id = exp_options[exp_idx]

    # --- select group ---
    groups = get_groups_for_experiment(exp_id, data)
    if not groups:
        raise RuntimeError(
            f"No groups assigned for experiment '{exp_id}' in config/experiments.yaml"
        )

    group_descriptions = []
    assignments = data.get("assignments", {}).get(exp_id, {})
    for gid in groups:
        members = [
            name for name, info in assignments.items()
            if info.get("group_id") == gid
        ]
        group_descriptions.append(", ".join(members) if members else "")

    group_idx = interactive_menu(
        "Select Group", groups, group_descriptions, prompt_enter=False,
    )
    group_id = groups[group_idx]

    return exp_id, group_id
