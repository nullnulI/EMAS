"""
MAP-THOR-only copy of the EMAS hybrid entry point.

This module deliberately does not modify ``scripts/hybrid_decision_loop.py``.
It reuses that module's planning helpers while replacing its environment
initialization and direct AI2-THOR adapter with benchmark-aware versions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator

from agents.skill_plan import execute_allocated_skills
from benchmark.action_log import BenchmarkActionLogger
from benchmark.config_loader import select_episode
from benchmark.episode import MapThorEpisode
from benchmark.evaluator import evaluate_official_checker
from benchmark.official import (
    initialize_episode,
    resolve_mapthor_root,
    snapshot_objects,
)
from planning.datagen import generation as gen
from scripts import hybrid_decision_loop as hybrid


DEFAULT_CLOUD_RENDERING_COMMIT_ID = "f0825767cd50d69f666c7f282e54abfe58f1e917"
DEFAULT_CLOUD_RENDERING_BUILD_NAME = (
    f"thor-CloudRendering-{DEFAULT_CLOUD_RENDERING_COMMIT_ID}"
)
DEFAULT_CLOUD_RENDERING_ZIP = (
    Path(__file__).resolve().parents[1] / f"{DEFAULT_CLOUD_RENDERING_BUILD_NAME}.zip"
)
EMAS_ROOT = Path(__file__).resolve().parents[1]
AGENTS_ROOT = EMAS_ROOT / "agents"
PROJECT_AI2THOR_ROOT = EMAS_ROOT / "ai2thor"
PROJECT_AI2THOR_RELEASES_DIR = PROJECT_AI2THOR_ROOT / "releases"
RUNTIME_AI2THOR_ROOT = Path(os.environ.get("EMAS_BENCHMARK_AI2THOR_CACHE", "/tmp/emas_ai2thor"))
RUNTIME_AI2THOR_RELEASES_DIR = RUNTIME_AI2THOR_ROOT / "releases"


def _separate_agent_zero_on_reachable_position(controller: Any) -> None:
    """Move agent 0 to a valid point away from the other agents, when possible."""

    event = controller.step(action="GetReachablePositions", agentId=0)
    metadata = getattr(event, "metadata", {}) or {}
    if not metadata.get("lastActionSuccess", True):
        return
    positions = metadata.get("actionReturn") or []
    candidates = [
        position
        for position in positions
        if isinstance(position, dict)
        and isinstance(position.get("x"), (int, float))
        and isinstance(position.get("z"), (int, float))
    ]
    if not candidates:
        return

    other_positions = []
    for agent_event in getattr(controller.last_event, "events", []) or []:
        agent = (getattr(agent_event, "metadata", {}) or {}).get("agent") or {}
        position = agent.get("position") or {}
        if isinstance(position.get("x"), (int, float)) and isinstance(
            position.get("z"), (int, float)
        ):
            other_positions.append(position)

    if other_positions:
        target = max(
            candidates,
            key=lambda candidate: min(
                math.hypot(
                    float(candidate["x"]) - float(other["x"]),
                    float(candidate["z"]) - float(other["z"]),
                )
                for other in other_positions
            ),
        )
    else:
        target = candidates[0]

    controller.step(
        action="Teleport",
        position=target,
        rotation={"x": 0, "y": 270, "z": 0},
        agentId=0,
        forceAction=True,
    )


class BenchmarkAI2ThorDirectAdapter(hybrid.CommunicationAdapter):
    """Direct adapter that attaches the official checker to real actions."""

    def __init__(
        self,
        controller: Any,
        args: argparse.Namespace,
        action_logger: BenchmarkActionLogger,
    ) -> None:
        self.controller = controller
        self.args = args
        self.action_logger = action_logger

    def send_task_allocation(self, payload: dict[str, Any], loop_dir: Path) -> None:
        hybrid.save_json(payload, loop_dir / "communication_outbox.json")

    @staticmethod
    def _merge_executions(parts: list[dict[str, Any]]) -> dict[str, Any]:
        traces = [trace for part in parts for trace in part.get("traces") or []]
        completed = [
            task_id for part in parts for task_id in part.get("completed_task_ids") or []
        ]
        collided_objects = sorted(
            {
                str(object_id)
                for part in parts
                for object_id in part.get("collided_objects") or []
            }
        )
        simulation_values = [
            float(part["simulation_time_seconds"])
            for part in parts
            if isinstance(part.get("simulation_time_seconds"), (int, float))
        ]
        return {
            "completed_task_ids": completed,
            "traces": traces,
            "execution_time_seconds": sum(
                float(part.get("execution_time_seconds") or 0.0) for part in parts
            ),
            "macro_step_wall_time_seconds": sum(
                float(part.get("macro_step_wall_time_seconds") or 0.0) for part in parts
            ),
            "simulation_time_seconds": sum(simulation_values) if simulation_values else None,
            "collided": any(bool(part.get("collided")) for part in parts),
            "collision_count": sum(int(part.get("collision_count") or 0) for part in parts),
            "collided_objects": collided_objects,
        }

    def receive_execution_report(
        self,
        payload: dict[str, Any],
        loop_dir: Path,
    ) -> dict[str, Any]:
        parts = []
        macro_step = int(payload.get("loop_index", 0)) + 1
        for assignment in payload.get("assignments") or []:
            subtask = assignment.get("subtask") or {}
            self.action_logger.bind(
                macro_step=macro_step,
                subtask_id=str(subtask.get("id")),
                agent_id=int(assignment.get("agent_id", 0)),
            )
            parts.append(
                execute_allocated_skills(
                    self.controller,
                    [assignment],
                    use_qwen=not self.args.disable_qwen
                    and not self.args.disable_skill_qwen,
                    qwen_model_path=self.args.planning_model_path,
                    qwen_conv_mode=self.args.qwen_conv_mode,
                    qwen_num_gpus=self.args.qwen_num_gpus,
                    max_steps=self.args.skill_max_steps,
                    complete_on_execute=self.args.skill_complete_on_execute,
                    step_callback=self.action_logger,
                )
            )
        report = {"execution": self._merge_executions(parts)}
        hybrid.save_json(report, loop_dir / "communication_inbox.json")
        return report


def _install_scene_object_catalog(args: argparse.Namespace, objects: list[dict[str, Any]]) -> None:
    """Attach an episode-local catalogue without exposing it in serialized CLI args."""

    args._scene_object_catalog = deepcopy(objects)


def _save_object_snapshot(objects: list[dict[str, Any]], path: Path) -> None:
    fields = {
        "objectId",
        "objectType",
        "name",
        "position",
        "parentReceptacles",
        "receptacleObjectIds",
        "isOpen",
        "isToggled",
        "isDirty",
        "isSliced",
        "isCooked",
        "isFilledWithLiquid",
        "fillLiquid",
        "visible",
    }
    compact = [{key: value for key, value in obj.items() if key in fields} for obj in objects]
    hybrid.save_json(compact, path)


def _prepare_episode_output(episode_dir: Path) -> None:
    """Remove only artifacts owned by a previous attempt of this episode."""

    for directory_name in ("initial_planning", "loops", "task_execution", "runtime", "services", "videos"):
        directory = episode_dir / directory_name
        if directory.exists():
            shutil.rmtree(directory)
    for file_name in (
        "action_log.jsonl",
        "args.json",
        "benchmark_evaluation.json",
        "benchmark_result.json",
        "error.json",
        "final_object_state.json",
        "initial_agent_states.json",
        "initial_object_state.json",
        "initial_scenegraph_record.json",
        "planning_failure.json",
        "run_record.json",
        "runtime_task_graph_final.json",
        "runtime_task_graph_initial.json",
    ):
        (episode_dir / file_name).unlink(missing_ok=True)


@contextmanager
def _use_project_ai2thor_release_cache() -> Iterator[None]:
    """Point AI2-THOR release lookup at the repo-local ai2thor directory."""

    original_releases_dir = gen.Controller.releases_dir
    RUNTIME_AI2THOR_RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    (RUNTIME_AI2THOR_ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    gen.Controller.releases_dir = property(lambda self: str(RUNTIME_AI2THOR_RELEASES_DIR))
    try:
        yield
    finally:
        gen.Controller.releases_dir = original_releases_dir



def _find_available_port(start_port: int) -> int:
    for port in range(int(start_port), int(start_port) + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"no free localhost port found from {start_port}")


def _wait_http_health(
    url: str,
    *,
    timeout_seconds: float = 90.0,
    required_status: str | None = None,
) -> dict[str, Any]:
    from urllib.request import urlopen

    deadline = time.time() + float(timeout_seconds)
    last_error = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                if required_status is not None and parsed.get("status") != required_status:
                    last_error = RuntimeError(
                        f"service returned status {parsed.get('status')!r}; expected {required_status!r}: {parsed}"
                    )
                    time.sleep(0.5)
                    continue
                return parsed
            return {"status": "ok", "raw": parsed}
        except Exception as exc:  # Service may still be starting.
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"service health check timed out for {url}: {last_error}")


def _receiver_from_existing_controller(controller: Any, episode: MapThorEpisode, action_logger: BenchmarkActionLogger):
    from agents import ai2thor_receiver_server as receiver

    thor = receiver.NativeControllerThorServer.__new__(receiver.NativeControllerThorServer)
    thor.scene = episode.floorplan
    thor.robot_count = max(1, int(episode.agent_count))
    thor.headless = True
    thor.width = 600
    thor.height = 600
    thor.robot0_dx = 0.0
    thor.robot0_dz = 0.0
    thor.robot0_left = 0.0
    thor.robot0_right = 0.0
    thor.robot0_back = 0.0
    thor.robot0_dyaw = 0.0
    thor.robot0_at_fridge = False
    thor.robot0_fridge_distance = 1.0
    thor.controller = controller
    thor.robots = []
    thor.lock = threading.RLock()
    thor._step_count = 0
    thor._window_names = {}
    thor.action_callback = action_logger
    thor.render_action_frames = False

    event = getattr(controller, "last_event", None)
    for robot_id in range(thor.robot_count):
        agent_event = thor._event_for_robot(event, robot_id) if event is not None else None
        meta = dict(getattr(agent_event, "metadata", {}) or {})
        agent = dict(meta.get("agent") or {})
        robot = receiver.RobotState(
            robot_id=robot_id,
            name=f"Robot{robot_id}",
            position=dict(agent.get("position") or {"x": 0.0, "y": receiver.DEFAULT_AGENT_Y, "z": 0.0}),
            rotation=dict(agent.get("rotation") or {"x": 0.0, "y": 0.0, "z": 0.0}),
            horizon=float(agent.get("cameraHorizon") or 0.0),
            task=f"MAP-THOR benchmark agent slot {robot_id}",
            last_event=agent_event,
        )
        thor._update_robot_pose_from_metadata(robot, meta)
        thor.robots.append(robot)
    return receiver, thor


@contextmanager
def _managed_benchmark_receiver(
    args: argparse.Namespace,
    episode_dir: Path,
    controller: Any,
    episode: MapThorEpisode,
    action_logger: BenchmarkActionLogger,
) -> Iterator[dict[str, Any]]:
    from http.server import ThreadingHTTPServer
    from agents import ai2thor_receiver_server as receiver

    services_dir = episode_dir / "services"
    services_dir.mkdir(parents=True, exist_ok=True)
    log_path = services_dir / "receiver.log"
    port = _find_available_port(int(getattr(args, "receiver_port_base", 19010)))
    original_thor = receiver.thor_instance
    original_log_event = receiver.log_event
    receiver_module, thor = _receiver_from_existing_controller(controller, episode, action_logger)
    thor.render_action_frames = bool(getattr(args, "save_agent_video", False))

    log_handle = log_path.open("a", encoding="utf-8")

    def log_event_to_file(channel: str, message: str = "", *, file=None, blank_before: bool = False):
        if os.environ.get("EMAS_BENCHMARK_RECEIVER_STDOUT") == "1":
            original_log_event(channel, message, file=file, blank_before=blank_before)
        if blank_before:
            print("", file=log_handle, flush=True)
        for line in (str(message).splitlines() or [""]):
            print(f"[{channel}] {line}", file=log_handle, flush=True)

    receiver.thor_instance = thor
    receiver.log_event = log_event_to_file
    server = ThreadingHTTPServer(("127.0.0.1", port), receiver_module.NativeControllerReceiverHandler)
    thread = threading.Thread(target=server.serve_forever, name="mapthor-benchmark-receiver", daemon=True)
    thread.start()
    info = {
        "receiver_url": f"http://127.0.0.1:{port}",
        "execute_actions_url": f"http://127.0.0.1:{port}/execute_actions",
        "health_url": f"http://127.0.0.1:{port}/health",
        "log_path": str(log_path),
        "port": port,
    }
    try:
        _wait_http_health(info["health_url"], timeout_seconds=10.0)
        yield info
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)
        receiver.thor_instance = original_thor
        receiver.log_event = original_log_event
        log_handle.close()


def _task_execution_server_command(args: argparse.Namespace, *, receiver_url: str, port: int, output_dir: Path) -> list[str]:
    python_bin = getattr(args, "task_service_python", None) or os.environ.get("TASK_SERVICE_PYTHON") or sys.executable
    return [
        str(python_bin),
        str(AGENTS_ROOT / "task_execution_server.py"),
        "--host",
        "127.0.0.1",
        "--port",
        str(int(port)),
        "--receiver-url",
        receiver_url,
        "--model-path",
        str(getattr(args, "task_service_model_path", "/225010231/mwl/Linhao/models/Qwen3.5-4B")),
        "--device",
        str(getattr(args, "task_service_device", "cuda")),
        "--device-map",
        str(getattr(args, "task_service_device_map", "auto")),
        "--dtype",
        str(getattr(args, "task_service_dtype", "float16")),
        "--max-new-tokens",
        str(int(getattr(args, "max_new_tokens", 512) or 512)),
        "--output",
        str(output_dir.expanduser().resolve()),
        "--max-replan-steps",
        str(int(getattr(args, "task_service_max_replan_steps", 10))),
        "--relay-agent-max-turns",
        str(int(getattr(args, "task_service_relay_agent_max_turns", 8))),
        "--max-actions",
        str(int(getattr(args, "task_service_max_actions", 8))),
    ]


@contextmanager
def _managed_task_execution_server(
    args: argparse.Namespace,
    episode_dir: Path,
    *,
    receiver_url: str,
    wait_for_receiver: bool = True,
) -> Iterator[dict[str, Any]]:
    services_dir = episode_dir / "services"
    services_dir.mkdir(parents=True, exist_ok=True)
    port = _find_available_port(int(getattr(args, "task_service_port_base", 18090)))
    task_service_url = f"http://127.0.0.1:{port}/execute_task"
    health_url = f"http://127.0.0.1:{port}/health"
    log_path = services_dir / "task_execution_server.log"
    output_dir = episode_dir / "task_execution"
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = _task_execution_server_command(args, receiver_url=receiver_url, port=port, output_dir=output_dir)
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    agent_cuda_devices = str(
        getattr(
            args,
            "task_service_cuda_visible_devices",
            os.environ.get("EMAS_AGENTS_CUDA_VISIBLE_DEVICES", "1"),
        )
    ).strip()
    if agent_cuda_devices:
        env["CUDA_VISIBLE_DEVICES"] = agent_cuda_devices
    python_bin = Path(cmd[0])
    conda_prefix = python_bin.parent.parent if python_bin.name.startswith("python") else None
    if conda_prefix is not None and (conda_prefix / "lib").is_dir():
        env["CONDA_PREFIX"] = str(conda_prefix)
        env["LD_LIBRARY_PATH"] = str(conda_prefix / "lib") + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    log_handle = log_path.open("ab")
    process = subprocess.Popen(cmd, cwd=str(AGENTS_ROOT), stdout=log_handle, stderr=subprocess.STDOUT, env=env)
    info = {
        "task_service_url": task_service_url,
        "health_url": health_url,
        "log_path": str(log_path),
        "pid": process.pid,
        "command": cmd,
        "port": port,
        "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        health = _wait_http_health(
            health_url,
            timeout_seconds=180.0,
            required_status="ready" if wait_for_receiver else None,
        )
        if not health.get("backend_ready"):
            raise RuntimeError(f"task execution backend did not load successfully: {health}")
        yield info
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10.0)
        log_handle.close()


@contextmanager
def _managed_full_system_services(
    args: argparse.Namespace,
    episode_dir: Path,
    controller: Any,
    episode: MapThorEpisode,
    action_logger: BenchmarkActionLogger,
) -> Iterator[dict[str, Any]]:
    if args.execution_mode != "task_service" or not getattr(args, "task_service_autostart", False):
        yield {}
        return
    original_task_service_url = args.task_service_url
    with _managed_benchmark_receiver(args, episode_dir, controller, episode, action_logger) as receiver_info:
        shared_task_service = getattr(args, "_shared_task_service_info", None)
        if isinstance(shared_task_service, dict):
            expected_receiver_url = str(shared_task_service.get("receiver_url") or "")
            if expected_receiver_url and expected_receiver_url != receiver_info["receiver_url"]:
                raise RuntimeError(
                    "shared task service receiver URL mismatch: "
                    f"expected {expected_receiver_url}, got {receiver_info['receiver_url']}"
                )
            task_service_info = dict(shared_task_service)
            health_url = str(task_service_info.get("health_url") or "")
            if health_url:
                _wait_http_health(health_url, timeout_seconds=60.0, required_status="ready")
            args.task_service_url = task_service_info["task_service_url"]
            service_info = {
                "receiver": receiver_info,
                "task_execution_server": task_service_info,
                "shared_task_execution_server": True,
            }
            hybrid.save_json(service_info, episode_dir / "services" / "managed_services.json")
            try:
                yield service_info
            finally:
                args.task_service_url = original_task_service_url
            return
        with _managed_task_execution_server(args, episode_dir, receiver_url=receiver_info["receiver_url"]) as task_service_info:
            args.task_service_url = task_service_info["task_service_url"]
            service_info = {"receiver": receiver_info, "task_execution_server": task_service_info}
            hybrid.save_json(service_info, episode_dir / "services" / "managed_services.json")
            try:
                yield service_info
            finally:
                args.task_service_url = original_task_service_url


def _ensure_default_cloud_rendering_build(args: argparse.Namespace) -> None:
    """Use the vendored CloudRendering build unless the caller chose another build."""

    if args.platform != "cloud":
        return
    if args.commit_id or args.branch or args.local_build or args.local_executable_path:
        return

    args.commit_id = DEFAULT_CLOUD_RENDERING_COMMIT_ID
    release_dir = RUNTIME_AI2THOR_RELEASES_DIR / DEFAULT_CLOUD_RENDERING_BUILD_NAME
    project_release_dir = PROJECT_AI2THOR_RELEASES_DIR / DEFAULT_CLOUD_RENDERING_BUILD_NAME
    if not release_dir.exists() and project_release_dir.exists():
        release_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            release_dir.symlink_to(project_release_dir, target_is_directory=True)
        except FileExistsError:
            pass
    executable = release_dir / DEFAULT_CLOUD_RENDERING_BUILD_NAME
    data_dir = release_dir / f"{DEFAULT_CLOUD_RENDERING_BUILD_NAME}_Data"
    if executable.exists() and data_dir.is_dir():
        if not os.access(executable, os.X_OK):
            executable.chmod(0o755)
        return

    if not DEFAULT_CLOUD_RENDERING_ZIP.exists():
        return

    release_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DEFAULT_CLOUD_RENDERING_ZIP) as archive:
        archive.extractall(release_dir)
    if not os.access(executable, os.X_OK):
        executable.chmod(0o755)


@contextmanager
def _software_vulkan_controller_environment(
    args: argparse.Namespace,
) -> Iterator[None]:
    """Avoid forcing a CUDA-mapped Vulkan device when llvmpipe is requested."""

    use_software_vulkan = (
        args.platform == "cloud"
        and bool(getattr(args, "allow_software_vulkan", False))
    )
    if not use_software_vulkan:
        yield
        return

    missing = object()
    previous = os.environ.pop("CUDA_VISIBLE_DEVICES", missing)
    try:
        yield
    finally:
        if previous is missing:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(previous)


@contextmanager
def _benchmark_overrides(
    *,
    controller: Any,
    action_logger: BenchmarkActionLogger,
) -> Iterator[None]:
    """Temporarily replace only the extension points used by the copied runner."""

    original_initialize = hybrid.initialize_scenegraph_and_agents
    original_adapter = hybrid.AI2ThorDirectAdapter
    original_relay_adapter = hybrid.TaskExecutionServiceAdapter

    def initialize_scenegraph_and_agents(
        args: argparse.Namespace,
        run_dir: Path,
        skills_by_agent: dict[str, list[str]],
    ) -> tuple[Any, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        runtime_dir = run_dir / "runtime"
        initial_stage = gen.collect_stage(
            controller,
            runtime_dir,
            "benchmark_initial",
            args,
        )
        scenegraph_info = gen.build_stage_scenegraph(initial_stage, args)
        agent_states = hybrid.attach_agent_skills(
            initial_stage["agent_states"],
            skills_by_agent,
        )
        return controller, scenegraph_info, agent_states, {
            "source": "map_thor_official_initializer",
            "global_mapping": False,
            "initial_stage": {
                key: value for key, value in initial_stage.items() if key != "agent_states"
            },
            "scenegraph": scenegraph_info,
        }

    def adapter_factory(
        supplied_controller: Any,
        args: argparse.Namespace,
    ) -> BenchmarkAI2ThorDirectAdapter:
        return BenchmarkAI2ThorDirectAdapter(supplied_controller, args, action_logger)

    def task_service_adapter_factory(args: argparse.Namespace):
        adapter = original_relay_adapter(args)
        adapter.action_logger = action_logger
        return adapter

    hybrid.initialize_scenegraph_and_agents = initialize_scenegraph_and_agents
    hybrid.AI2ThorDirectAdapter = adapter_factory
    hybrid.TaskExecutionServiceAdapter = task_service_adapter_factory
    try:
        yield
    finally:
        hybrid.initialize_scenegraph_and_agents = original_initialize
        hybrid.AI2ThorDirectAdapter = original_adapter
        hybrid.TaskExecutionServiceAdapter = original_relay_adapter


def _make_initialized_controller(
    args: argparse.Namespace,
    episode: MapThorEpisode,
    mapthor_root: Path,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    args.scene_name = episode.floorplan
    args.task = episode.instruction
    args.agentnum = episode.agent_count
    args.max_task_loops = episode.timeout
    args.global_scenegraph = False
    if args.execution_mode != "task_service":
        args.execution_mode = "ai2thor"
    args.adapter_auto_success = False

    gen.start_xserver_if_needed(args)
    _ensure_default_cloud_rendering_build(args)
    if not args.allow_software_vulkan:
        gen.validate_cloud_rendering_environment(args)
    with _use_project_ai2thor_release_cache():
        with _software_vulkan_controller_environment(args):
            controller = gen.make_controller(args)
    gen.reset_scene(controller, args)

    # Keep the agents separated without assuming one coordinate is valid on every map.
    if episode.agent_count > 1:
        _separate_agent_zero_on_reachable_position(controller)

    checker = initialize_episode(controller, episode, mapthor_root)
    initial_objects = snapshot_objects(controller)
    return controller, checker, initial_objects


def _relay_snapshot_paths_in_trace_order(episode_dir: Path) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()
    for inbox_path in sorted((episode_dir / "loops").glob("loop_*/communication_inbox.json")):
        try:
            report = hybrid.load_json(inbox_path)
        except Exception:
            continue
        for trace in (report.get("execution") or {}).get("traces") or []:
            response = trace.get("task_service_response") if isinstance(trace, dict) else None
            raw_path = response.get("image_path") if isinstance(response, dict) else None
            if not raw_path:
                continue
            path = Path(raw_path).expanduser()
            if not path.is_absolute():
                path = (EMAS_ROOT / path).resolve()
            if path.exists() and path not in seen:
                paths.append(path)
                seen.add(path)
    if paths:
        return paths
    return sorted((episode_dir / "task_execution").glob("*_scene.jpg"))


def _write_relay_snapshot_video(episode_dir: Path, args: argparse.Namespace) -> tuple[str | None, str | None]:
    if not bool(getattr(args, "save_agent_video", False)):
        return None, None
    image_paths = _relay_snapshot_paths_in_trace_order(episode_dir)
    if not image_paths:
        return None, "no relay scene snapshots were produced"
    output_path = episode_dir / "videos" / "relay_scene_snapshots.mp4"
    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image

        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = gen.MP4VideoWriter(output_path, float(getattr(args, "fps", 4) or 4))
        target_size = None
        try:
            for image_path in image_paths:
                frame = np.asarray(imageio.imread(image_path))
                if frame.ndim != 3:
                    continue
                if frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                if target_size is None:
                    target_size = (frame.shape[1], frame.shape[0])
                elif (frame.shape[1], frame.shape[0]) != target_size:
                    frame = np.asarray(Image.fromarray(frame).resize(target_size))
                writer.append_data(frame)
        finally:
            writer.close()
        if output_path.exists() and output_path.stat().st_size > 0:
            return str(output_path), None
        return None, "relay snapshot video writer produced no output"
    except Exception as exc:
        return None, repr(exc)


def run_benchmark_episode(
    args: argparse.Namespace,
    episode: MapThorEpisode,
    *,
    mapthor_root: Path,
) -> dict[str, Any]:
    """Run one official MAP-THOR episode through the dedicated hybrid copy."""

    episode_dir = episode.result_dir(Path(args.output_dir))
    _prepare_episode_output(episode_dir)
    episode_dir.mkdir(parents=True, exist_ok=True)
    args.run_name = episode.episode_id
    hybrid.save_json(episode.to_dict(), episode_dir / "episode.json")

    controller = None
    checker = None
    action_logger = None
    video_recorder = None
    agent_video_path = None
    agent_video_error = None
    record: dict[str, Any] = {}
    runtime_error = None
    started = time.time()
    initial_objects: list[dict[str, Any]] = []
    if hasattr(args, "_scene_object_catalog"):
        delattr(args, "_scene_object_catalog")
    try:
        controller, checker, initial_objects = _make_initialized_controller(
            args, episode, mapthor_root
        )
        _save_object_snapshot(initial_objects, episode_dir / "initial_object_state.json")
        _install_scene_object_catalog(args, initial_objects)
        video_recorder = gen.SampleVideoRecorder(episode_dir, args)
        try:
            video_recorder.capture(controller.last_event, {"stage": "benchmark_initial"})
        except Exception as exc:
            agent_video_error = repr(exc)
        action_logger = BenchmarkActionLogger(
            episode_dir / "action_log.jsonl",
            checker,
            initial_objects,
            event_observer=video_recorder.capture,
        )
        with _benchmark_overrides(controller=controller, action_logger=action_logger):
            with _managed_full_system_services(args, episode_dir, controller, episode, action_logger):
                record = hybrid.run_hybrid_loop(args)
    except Exception as exc:
        runtime_error = repr(exc)
        hybrid.save_json(
            {"episode_id": episode.episode_id, "error": runtime_error},
            episode_dir / "error.json",
        )
    finally:
        final_objects = snapshot_objects(controller) if controller is not None else []
        if final_objects:
            _save_object_snapshot(final_objects, episode_dir / "final_object_state.json")
        if video_recorder is not None:
            try:
                if controller is not None:
                    video_recorder.capture(controller.last_event, {"stage": "benchmark_final"})
            except Exception as exc:
                agent_video_error = repr(exc)
            try:
                agent_video_path = video_recorder.close()
            except Exception as exc:
                agent_video_error = repr(exc)
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass

    if checker is None or action_logger is None:
        evaluation = {
            "success": 0,
            "success_rate": 0.0,
            "transport_rate": 0.0,
            "coverage": 0.0,
            "balance": 0.0,
            "steps": 0,
            "timeout": episode.timeout,
            "low_level_actions": 0,
            "collisions": 0,
            "internal_all_done": False,
            "internal_benchmark_agree": True,
            "termination_reason": "runtime_error",
            "runtime_error": runtime_error,
        }
    else:
        evaluation = evaluate_official_checker(
            checker,
            action_logger,
            internal_all_done=bool(record.get("all_done")),
            macro_steps=len(record.get("loops") or []),
            timeout=episode.timeout,
            runtime_error=runtime_error,
        )

    relay_video_path, relay_video_error = _write_relay_snapshot_video(episode_dir, args)
    result = {
        "episode": episode.to_dict(),
        "evaluation": evaluation,
        "hybrid_record": record,
        "elapsed_seconds": time.time() - started,
        "mapthor_root": str(mapthor_root),
        "artifacts": {
            "agent_operations_video": agent_video_path,
            "relay_scene_snapshots_video": relay_video_path,
            "agent_operations_video_error": agent_video_error,
            "relay_scene_snapshots_video_error": relay_video_error,
        },
    }
    hybrid.save_json(result, episode_dir / "benchmark_result.json")
    hybrid.save_json(evaluation, episode_dir / "benchmark_evaluation.json")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = hybrid.build_parser()
    parser.description = "Run the benchmark-only EMAS hybrid copy on MAP-THOR."
    for action in parser._actions:
        if action.dest in {"scene_name", "task"}:
            action.required = False
    parser.set_defaults(
        output_dir=Path("benchmark/results"),
        execution_mode="ai2thor",
        global_scenegraph=False,
        adapter_auto_success=False,
    )
    parser.add_argument("--mapthor-root", type=Path, default=None)
    parser.add_argument("--mapthor-config-dir", type=Path, default=None)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--mapthor-task-id", type=int)
    selector.add_argument("--mapthor-task-name")
    parser.add_argument("--mapthor-floorplan-index", type=int, default=0)
    parser.add_argument(
        "--initialize-only",
        action="store_true",
        help="Apply the official initializer, save object state, and skip EMAS.",
    )
    parser.add_argument(
        "--allow-software-vulkan",
        action="store_true",
        help=(
            "Benchmark-only escape hatch for explicitly testing CloudRendering with "
            "a CPU Vulkan implementation such as llvmpipe. This may be slow or unstable."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    mapthor_root = resolve_mapthor_root(args.mapthor_root)
    config_dir = args.mapthor_config_dir or mapthor_root / "configs"
    episode = select_episode(
        config_dir,
        task_id=args.mapthor_task_id,
        task_name=args.mapthor_task_name,
        floorplan_index=args.mapthor_floorplan_index,
        seed=args.seed,
        agent_count=args.agentnum,
        tasks_root=mapthor_root / "AI2Thor" / "Tasks",
    )
    if args.initialize_only:
        controller, _, objects = _make_initialized_controller(args, episode, mapthor_root)
        output_dir = episode.result_dir(Path(args.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)
        hybrid.save_json(episode.to_dict(), output_dir / "episode.json")
        _save_object_snapshot(objects, output_dir / "initial_object_state.json")
        controller.stop()
        print(json.dumps({"episode_id": episode.episode_id, "objects": len(objects)}, indent=2))
        return

    result = run_benchmark_episode(args, episode, mapthor_root=mapthor_root)
    print(
        json.dumps(
            {
                "episode_id": episode.episode_id,
                "evaluation": result["evaluation"],
                "result_dir": str(episode.result_dir(Path(args.output_dir))),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
