from __future__ import annotations

from argparse import Namespace
from contextlib import contextmanager
import os
from pathlib import Path

from benchmark.action_log import BenchmarkActionLogger
from scripts import benchmark_hybrid_decision_loop as benchmark_hybrid


class FakeController:
    def __init__(self):
        self.last_event = None
        self.steps = []

    def step(self, **kwargs):
        self.steps.append(kwargs)
        return self.last_event


class FakeEvent:
    def __init__(self, metadata=None, events=None):
        self.metadata = metadata or {}
        self.events = events or []


class FakeEpisode:
    floorplan = "FloorPlan1"
    instruction = "open the fridge"
    agent_count = 2
    timeout = 5


def test_benchmark_parser_accepts_full_system_task_service_args():
    args = benchmark_hybrid.build_parser().parse_args([
        "--mapthor-task-id", "1",
        "--execution-mode", "task_service",
        "--task-service-autostart",
        "--task-service-python", "/opt/qwen/bin/python",
        "--task-service-model-path", "/models/qwen",
        "--task-service-device", "cpu",
        "--task-service-cuda-visible-devices", "3",
        "--task-service-port-base", "18100",
        "--receiver-port-base", "19100",
    ])

    assert args.execution_mode == "task_service"
    assert args.task_service_autostart is True
    assert args.task_service_python == "/opt/qwen/bin/python"
    assert args.task_service_model_path == "/models/qwen"
    assert args.task_service_device == "cpu"
    assert args.task_service_cuda_visible_devices == "3"
    assert args.task_service_port_base == 18100
    assert args.receiver_port_base == 19100


def test_agent_zero_spawn_is_selected_from_reachable_positions():
    controller = FakeController()
    agent_one = FakeEvent({"agent": {"position": {"x": 0.0, "y": 0.9, "z": 0.0}}})
    controller.last_event = FakeEvent(events=[agent_one])
    reachable_event = FakeEvent(
        {
            "lastActionSuccess": True,
            "actionReturn": [
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 2.0, "y": 0.9, "z": 1.0},
            ],
        }
    )

    def step(**kwargs):
        controller.steps.append(kwargs)
        return reachable_event if kwargs["action"] == "GetReachablePositions" else FakeEvent()

    controller.step = step
    benchmark_hybrid._separate_agent_zero_on_reachable_position(controller)

    assert controller.steps[1]["action"] == "Teleport"
    assert controller.steps[1]["position"] == {"x": 2.0, "y": 0.9, "z": 1.0}


def test_agent_zero_spawn_stays_put_when_no_reachable_positions():
    controller = FakeController()
    controller.last_event = FakeEvent(
        {"lastActionSuccess": True, "actionReturn": []}, events=[]
    )

    benchmark_hybrid._separate_agent_zero_on_reachable_position(controller)

    assert controller.steps == [{"action": "GetReachablePositions", "agentId": 0}]


def test_make_initialized_controller_preserves_task_service_mode(monkeypatch):
    controller = FakeController()
    args = Namespace(
        execution_mode="task_service",
        platform="cloud",
        commit_id="existing",
        branch=None,
        local_build=None,
        local_executable_path=None,
        allow_software_vulkan=True,
    )
    episode = FakeEpisode()

    monkeypatch.setattr(benchmark_hybrid.gen, "start_xserver_if_needed", lambda unused_args: None)
    monkeypatch.setattr(benchmark_hybrid.gen, "validate_cloud_rendering_environment", lambda unused_args: None)
    controller_cuda_visibility = []

    def make_controller(unused_args):
        controller_cuda_visibility.append(os.environ.get("CUDA_VISIBLE_DEVICES"))
        return controller

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(benchmark_hybrid.gen, "make_controller", make_controller)
    monkeypatch.setattr(benchmark_hybrid.gen, "reset_scene", lambda unused_controller, unused_args: None)
    monkeypatch.setattr(benchmark_hybrid, "initialize_episode", lambda *unused: "checker")
    monkeypatch.setattr(benchmark_hybrid, "snapshot_objects", lambda unused_controller: [])

    returned_controller, checker, objects = benchmark_hybrid._make_initialized_controller(
        args, episode, Path("/tmp/mapthor")
    )

    assert returned_controller is controller
    assert checker == "checker"
    assert objects == []
    assert args.execution_mode == "task_service"
    assert controller_cuda_visibility == [None]
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_benchmark_overrides_injects_action_logger_into_relay_adapter(monkeypatch):
    class FakeRelayAdapter:
        def __init__(self, args):
            self.args = args

    action_logger = object()
    monkeypatch.setattr(benchmark_hybrid.hybrid, "TaskExecutionServiceAdapter", FakeRelayAdapter)

    with benchmark_hybrid._benchmark_overrides(controller=None, action_logger=action_logger):
        adapter = benchmark_hybrid.hybrid.TaskExecutionServiceAdapter(Namespace())
        assert adapter.action_logger is action_logger


def test_task_execution_server_command_uses_managed_receiver_and_model_path(tmp_path):
    args = Namespace(
        task_service_python="/opt/qwen/bin/python",
        task_service_model_path="/models/qwen",
        task_service_device="cpu",
        task_service_device_map="auto",
        task_service_dtype="float16",
        max_new_tokens=123,
        task_service_max_replan_steps=4,
        task_service_relay_agent_max_turns=5,
        task_service_max_actions=6,
    )

    cmd = benchmark_hybrid._task_execution_server_command(
        args, receiver_url="http://127.0.0.1:19100", port=18100, output_dir=tmp_path
    )

    assert cmd[0] == "/opt/qwen/bin/python"
    assert str(benchmark_hybrid.AGENTS_ROOT / "task_execution_server.py") in cmd
    assert cmd[cmd.index("--receiver-url") + 1] == "http://127.0.0.1:19100"
    assert cmd[cmd.index("--port") + 1] == "18100"
    assert cmd[cmd.index("--model-path") + 1] == "/models/qwen"
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--output") + 1] == str(tmp_path)


def test_task_execution_server_command_resolves_relative_output_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = Namespace(
        task_service_python="python",
        task_service_model_path="/models/qwen",
        task_service_device="cpu",
        task_service_device_map="auto",
        task_service_dtype="float16",
        max_new_tokens=64,
        task_service_max_replan_steps=4,
        task_service_relay_agent_max_turns=5,
        task_service_max_actions=6,
    )
    cmd = benchmark_hybrid._task_execution_server_command(
        args,
        receiver_url="http://127.0.0.1:19100",
        port=18100,
        output_dir=Path("relative/task_execution"),
    )

    assert cmd[cmd.index("--output") + 1] == str((tmp_path / "relative/task_execution").resolve())


def test_managed_services_reuses_shared_task_service_and_restores_url(tmp_path, monkeypatch):
    receiver_url = "http://127.0.0.1:19100"
    task_service_url = "http://127.0.0.1:18100/execute_task"
    args = Namespace(
        execution_mode="task_service",
        task_service_autostart=True,
        task_service_url="http://original/execute_task",
        _shared_task_service_info={
            "receiver_url": receiver_url,
            "task_service_url": task_service_url,
            "health_url": "http://127.0.0.1:18100/health",
        },
    )
    health_calls = []

    @contextmanager
    def fake_receiver(*unused_args, **unused_kwargs):
        yield {"receiver_url": receiver_url}

    monkeypatch.setattr(benchmark_hybrid, "_managed_benchmark_receiver", fake_receiver)
    monkeypatch.setattr(
        benchmark_hybrid,
        "_wait_http_health",
        lambda url, **kwargs: health_calls.append((url, kwargs)) or {"status": "ready"},
    )
    monkeypatch.setattr(benchmark_hybrid.hybrid, "save_json", lambda *unused: None)

    with benchmark_hybrid._managed_full_system_services(
        args, tmp_path, object(), FakeEpisode(), object()
    ) as service_info:
        assert args.task_service_url == task_service_url
        assert service_info["shared_task_execution_server"] is True

    assert args.task_service_url == "http://original/execute_task"
    assert health_calls[0][1]["required_status"] == "ready"


def test_relay_snapshot_paths_follow_loop_trace_order(tmp_path):
    first = tmp_path / "task_execution" / "loop_0_T1_scene.jpg"
    second = tmp_path / "task_execution" / "loop_1_T5_scene.jpg"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    for loop_index, image_path in ((0, first), (1, second)):
        inbox = tmp_path / "loops" / f"loop_{loop_index:03d}" / "communication_inbox.json"
        inbox.parent.mkdir(parents=True)
        benchmark_hybrid.hybrid.save_json(
            {
                "execution": {
                    "traces": [{"task_service_response": {"image_path": str(image_path)}}]
                }
            },
            inbox,
        )

    assert benchmark_hybrid._relay_snapshot_paths_in_trace_order(tmp_path) == [first, second]


def test_action_logger_maps_ai2thor_pickup_to_official_pick_name(tmp_path):
    class Checker:
        subtasks_completed = []

        def __init__(self):
            self.subtasks = ["PickObject(Book_1)"]
            self.actions = []

        def perform_metric_check(self, action, success, inventory):
            self.actions.append((action, success, inventory))

    class Event:
        metadata = {
            "inventoryObjects": [{"objectId": "Book|0|0|0"}],
        }

    checker = Checker()
    logger = BenchmarkActionLogger(
        tmp_path / "action_log.jsonl",
        checker,
        [{"objectId": "Book|0|0|0", "objectType": "Book"}],
    )
    logger(
        Event(),
        {
            "action": "PickupObject",
            "params": {"objectId": "Book|0|0|0"},
            "lastActionSuccess": True,
            "agent_id": 0,
        },
    )

    assert checker.actions == [("PickObject(Book_1)", True, "Book_1")]


def test_action_logger_video_observer_is_fail_soft(tmp_path):
    class Checker:
        subtasks_completed = []

        def perform_metric_check(self, *args):
            return None

    class Event:
        metadata = {"inventoryObjects": []}

    logger = BenchmarkActionLogger(
        tmp_path / "action_log.jsonl",
        Checker(),
        [],
        event_observer=lambda *args: (_ for _ in ()).throw(RuntimeError("video unavailable")),
    )
    logger(Event(), {"action": "MoveAhead", "lastActionSuccess": True, "agent_id": 0})

    assert len(logger.records) == 1


def test_prepare_episode_output_removes_only_previous_attempt_artifacts(tmp_path):
    episode_dir = tmp_path / "episode"
    (episode_dir / "loops" / "loop_999").mkdir(parents=True)
    (episode_dir / "services").mkdir()
    (episode_dir / "loops" / "loop_999" / "old.json").write_text("{}", encoding="utf-8")
    (episode_dir / "services" / "receiver.log").write_text("old", encoding="utf-8")
    (episode_dir / "benchmark_result.json").write_text("{}", encoding="utf-8")
    unrelated = episode_dir / "user_note.txt"
    unrelated.write_text("keep", encoding="utf-8")

    benchmark_hybrid._prepare_episode_output(episode_dir)

    assert not (episode_dir / "loops").exists()
    assert not (episode_dir / "services").exists()
    assert not (episode_dir / "benchmark_result.json").exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"
