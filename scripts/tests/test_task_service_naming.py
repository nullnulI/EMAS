from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _mainline_files() -> list[Path]:
    files = [REPO_ROOT / "README.md"]
    for directory, patterns in (
        (REPO_ROOT / "agents", ("*.py", "*.sh", "*.md")),
        (REPO_ROOT / "scripts", ("*.py", "*.sh", "*.md")),
        (REPO_ROOT / "benchmark", ("*.py", "*.sh", "*.md")),
    ):
        for pattern in patterns:
            files.extend(directory.glob(pattern))
    return sorted({path for path in files if path.is_file()})


def test_mainline_uses_task_execution_service_names_only():
    forbidden = (
        "relay_" + "task_server",
        "Relay" + "TaskServiceAdapter",
        "Relay" + "TaskService",
        "Relay" + "RuntimeConfig",
        "run_" + "relay_task_server",
        "--relay-" + "service-",
        "--relay-" + "task-",
        '"relay_' + 'service"',
        "'relay_" + "service'",
        '"relay_' + 'tasks"',
        "relay_" + "tasks/",
    )
    violations: list[str] = []
    for path in _mainline_files():
        text = path.read_text(encoding="utf-8")
        for legacy_name in forbidden:
            if legacy_name in text:
                violations.append(f"{path.relative_to(REPO_ROOT)}: {legacy_name}")
    assert violations == []


def test_internal_relay_agent_names_remain_supported():
    relay_agent = REPO_ROOT / "agents" / "EmbodiedGPT_Pytorch" / "demo" / "relay_agent.py"
    text = relay_agent.read_text(encoding="utf-8")
    assert "class RelayAgentConfig" in text
    assert "def run_relay_agent" in text
    assert "relay_agent_max_turns" in (
        REPO_ROOT / "agents" / "task_execution_server.py"
    ).read_text(encoding="utf-8")
