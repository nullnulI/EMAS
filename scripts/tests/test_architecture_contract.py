from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _active_python_files():
    ignored_parts = {
        ".git",
        ".pytest_cache",
        "__pycache__",
        "benchmark/results",
        "recovery_snapshots",
    }
    for top_level in ("planning", "scripts", "benchmark", "agents"):
        for path in (ROOT / top_level).rglob("*.py"):
            if path == Path(__file__).resolve():
                continue
            relative = path.relative_to(ROOT).as_posix()
            if any(relative == item or relative.startswith(f"{item}/") for item in ignored_parts):
                continue
            yield path


def test_only_canonical_planning_and_allocation_modules_exist():
    assert (ROOT / "planning/task_graph.py").is_file()
    assert (ROOT / "planning/task_allocation.py").is_file()
    assert not (ROOT / "planning/utils/task_graph.py").exists()
    assert not (ROOT / "planning/utils/task_allocation.py").exists()
    assert not (ROOT / "planning/subtask_to_plan.py").exists()


def test_active_python_has_no_legacy_planning_imports():
    forbidden = (
        "planning.utils.task_graph",
        "planning.utils.task_allocation",
        "from utils.task_allocation",
    )
    offenders = []
    for path in _active_python_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(value in text for value in forbidden):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []


def test_no_backup_sources_remain_in_active_tree():
    assert list(ROOT.rglob("*.orig")) == []


def test_architecture_declares_full_graph_and_typed_blocking_contract():
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    assert "complete active graph" in architecture
    assert '"state": "dispatchable | blocked"' in architecture
    assert "must never mutate the semantic Task Graph" in architecture
