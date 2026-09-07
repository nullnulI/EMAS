"""
Trajectory data generator for EMAS + AI2-THOR.

Each episode produces one sample folder. A sample contains:
- initial RGB-D observations and agent states
- initial scene graph, task-relevant subgraph, and decomposed task graph
- per-loop task allocation, execution trace, post states, post scene graph,
  merged scene graph reference, and task completion status

The current executor is intentionally a small low-level placeholder that moves
agents through ``agents.skill_plan``. Replace that module's executor when a
VLA/WAM policy is ready.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
import prior
from tqdm import tqdm
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
from conceptgraph.utils.ai2thor import get_scene

EMAS_ROOT = Path(__file__).resolve().parents[2]
AI2THOR_ROOT = EMAS_ROOT / "ai2thor"
CONCEPTGRAPH_ROOT = EMAS_ROOT / "memory" / "concept-graphs"
WORKSPACE_ROOT = EMAS_ROOT.parents[1]
for path in (WORKSPACE_ROOT, EMAS_ROOT, AI2THOR_ROOT, CONCEPTGRAPH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from planning.task_allocation import build_execution_plan, first_execution_unit
from planning.model_config import DEFAULT_PLANNING_MODEL_PATH
from planning.task_graph import (
    TaskPlanningError,
    decompose_task_to_graph,
    rebuild_task_graph_views,
    save_task_graph,
)
from planning.extract_subgraph import extract_subgraph
from planning.utils.task_status import FAILURE, SUCCESS, judge_task_status
from agents.skill_plan import execute_allocated_skills
from memory.graph_update import (
    build_incremental_observed_scenegraph,
    diff_objects,
    infer_relation_deltas_from_metadata,
    update_scenegraph_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate EMAS trajectory training data.")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--task", type=str, required=True, help="Natural-language task instruction.")
    parser.add_argument("--episode", type=int, default=1, help="Number of trajectory samples to collect.")
    parser.add_argument("--agentnum", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--field-of-view", type=float, default=90.0)
    parser.add_argument("--quality", default="Very Low")
    parser.add_argument(
        "--platform",
        choices=["default", "cloud"],
        default="cloud",
        help="AI2-THOR rendering platform. 'cloud' is offscreen Vulkan rendering and is the default.",
    )
    parser.add_argument("--server-timeout", type=float, default=30.0)
    parser.add_argument("--server-start-timeout", type=float, default=60.0)
    parser.add_argument("--x-display", default=None)
    parser.add_argument("--gpu-device", type=int, default=None)
    parser.add_argument("--commit-id", default=None)
    parser.add_argument("--branch", default=None)
    parser.add_argument("--local-build", action="store_true")
    parser.add_argument("--local-executable-path", default=None)

    parser.add_argument("--output-dir", type=Path, default=Path("/225010231/mwl/EMAS/planning/datagen/runs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--save-agent-video",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one RGB video per sample showing all agent views during low-level action execution.",
    )
    parser.add_argument("--save-depth-video", action="store_true")
    parser.add_argument("--full-metadata", action="store_true")
    parser.add_argument("--log-memory", action="store_true", help="Log Python RSS and PyTorch CUDA usage around heavy stages.")

    parser.add_argument("--dataset-config", type=Path, default=CONCEPTGRAPH_ROOT / "conceptgraph/dataset/dataconfigs/ai2thor/ai2thor.yaml")
    parser.add_argument("--disable-scenegraph", action="store_true", help="Skip ConceptGraphs pipeline and write placeholders.")
    parser.add_argument("--scenegraph-dry-run", action="store_true")
    parser.add_argument("--scenegraph-skip-existing", action="store_true")
    parser.add_argument("--scenegraph-stride", type=int, default=1)
    parser.add_argument("--scenegraph-device", default="cuda")
    parser.add_argument("--scenegraph-build-device", default="cuda:0")
    parser.add_argument("--scenegraph-obj-min-detections", type=int, default=1)
    parser.add_argument("--scenegraph-min-views-per-object", type=int, default=1)
    parser.add_argument(
        "--global-scenegraph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build one global scene graph before task execution using sampled AI2-THOR poses.",
    )
    parser.add_argument("--global-map-samples", type=int, default=4, help="Number of reachable positions for global mapping. Uniform sampling creates 8 views per position.")
    parser.add_argument("--global-map-sample-method", choices=["uniform", "random"], default="uniform")
    parser.add_argument("--global-map-agent-id", type=int, default=0)
    parser.add_argument("--class-set", default="ram", choices=["scene", "generic", "minimal", "tag2text", "ram", "none"])
    parser.add_argument("--vlm-backend", default="qwen", choices=["llava", "qwen"])
    parser.add_argument(
        "--qwen-model-path",
        default="/225010231/mwl/EMAS/Qwen2.5-VL-7B-Instruct",
        help="Vision-language model used to construct scene graphs.",
    )
    parser.add_argument(
        "--planning-model-path",
        default=DEFAULT_PLANNING_MODEL_PATH,
        help="Text model used for task graphs, allocation, and skill planning (default: Qwen3.5-9B).",
    )
    parser.add_argument("--qwen-conv-mode", default="v0_mmtag")
    parser.add_argument("--qwen-num-gpus", type=int, default=1)
    parser.add_argument(
        "--planning-max-new-tokens",
        type=int,
        default=2048,
        help="Maximum Qwen output tokens for task decomposition and allocation.",
    )
    parser.add_argument(
        "--planning-max-attempts",
        type=int,
        default=3,
        help="Maximum Qwen attempts for valid task decomposition and allocation output.",
    )
    parser.add_argument("--disable-qwen", action="store_true")

    parser.add_argument("--max-task-loops", type=int, default=20)
    parser.add_argument(
        "--max-task-graph-replans",
        type=int,
        default=3,
        help="Maximum runtime Planning replacements per sample.",
    )
    parser.add_argument(
        "--max-task-replans-per-source",
        type=int,
        default=1,
        help="Maximum replans for one graph-version/runtime-block fingerprint.",
    )
    parser.add_argument(
        "--task-graph-replan-max-attempts",
        type=int,
        default=3,
        help="Maximum Planning attempts during one runtime graph replacement.",
    )
    parser.add_argument("--steps-per-subtask", type=int, default=3)
    parser.add_argument("--skill-max-steps", type=int, default=6, help="Max low-level AI2-THOR actions per assigned subtask.")
    parser.add_argument("--disable-skill-qwen", action="store_true", help="Use heuristic skill plans instead of Qwen low-level plans.")
    parser.add_argument("--max-task-retries", type=int, default=3, help="Max number of recoverable retries before a task is marked as failure.")
    parser.add_argument(
        "--skill-complete-on-execute",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Legacy execution trace flag. Task completion is now decided by task_status verification after each loop.",
    )
    return parser


def log(message: str) -> None:
    print(f"[generation] {message}", flush=True)


def log_memory(label: str, args: argparse.Namespace) -> None:
    if not getattr(args, "log_memory", False):
        return

    parts = []
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                parts.append(f"rss={float(line.split()[1]) / 1024:.1f} MiB")
                break
    except OSError:
        pass

    try:
        import torch

        if torch.cuda.is_available():
            allocated = sum(torch.cuda.memory_allocated(index) for index in range(torch.cuda.device_count()))
            reserved = sum(torch.cuda.memory_reserved(index) for index in range(torch.cuda.device_count()))
            parts.append(f"cuda_allocated={allocated / 2**20:.1f} MiB")
            parts.append(f"cuda_reserved={reserved / 2**20:.1f} MiB")
    except ImportError:
        pass

    log(f"memory {label}: {', '.join(parts) if parts else 'unavailable'}")


def display_is_reachable(display: str | None) -> bool:
    if not display:
        return False

    xdpyinfo = shutil.which("xdpyinfo")
    if xdpyinfo is not None:
        try:
            return (
                subprocess.run(
                    [xdpyinfo, "-display", display],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2.0,
                    check=False,
                ).returncode
                == 0
            )
        except subprocess.TimeoutExpired:
            return False

    if display.startswith(":"):
        display_number = display[1:].split(".", 1)[0]
        return Path(f"/tmp/.X11-unix/X{display_number}").exists()
    return False


def choose_xvfb_display() -> str:
    for display_number in range(1, 20):
        if not Path(f"/tmp/.X11-unix/X{display_number}").exists():
            return f":{display_number}"
    return ":99"


def start_xvfb() -> bool:
    xvfb = shutil.which("Xvfb")
    if xvfb is None:
        return False

    display = choose_xvfb_display()
    subprocess.Popen(
        [xvfb, display, "-screen", "0", "1024x768x24", "-ac", "+extension", "GLX", "+render", "-noreset"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    os.environ["DISPLAY"] = display
    for _ in range(30):
        if display_is_reachable(display):
            log(f"started Xvfb on DISPLAY={display}")
            return True
        time.sleep(0.1)
    log(f"started Xvfb on DISPLAY={display}, but it did not become reachable")
    return False


def start_xserver_if_needed(args: argparse.Namespace) -> None:
    if args.platform == "cloud":
        return

    if args.x_display:
        display = str(args.x_display)
        os.environ["DISPLAY"] = display
        if not display_is_reachable(display):
            raise RuntimeError(
                f"--x-display was set to {display}, but that display is not reachable. "
                "Start a valid X11/GLX server first, or remove --x-display and use --platform cloud."
            )
        return

    env_display = os.environ.get("DISPLAY")
    if env_display and display_is_reachable(env_display):
        return
    if env_display:
        log(f"ignoring stale DISPLAY={env_display}; no X server is reachable there")
        os.environ.pop("DISPLAY", None)

    log("no reachable X display found; switching to offscreen CloudRendering")
    args.platform = "cloud"


def validate_cloud_rendering_environment(args: argparse.Namespace) -> None:
    if args.platform != "cloud":
        return

    system_vulkaninfo = Path("/usr/bin/vulkaninfo")
    vulkaninfo = str(system_vulkaninfo) if system_vulkaninfo.is_file() else shutil.which("vulkaninfo")
    if vulkaninfo is None:
        raise RuntimeError(
            "CloudRendering requires Vulkan, but vulkaninfo is not installed. "
            "Install vulkan-tools/libvulkan1 or use a valid X11 platform."
        )

    vulkan_env = os.environ.copy()
    for name in (
        "VK_ADD_DRIVER_FILES",
        "VK_INSTANCE_LAYERS",
        "VK_LAYER_PATH",
        "VK_LOADER_DRIVERS_DISABLE",
        "VK_LOADER_DRIVERS_SELECT",
    ):
        vulkan_env.pop(name, None)

    diagnostics = []
    for attempt in range(1, 4):
        result = subprocess.run(
            [vulkaninfo, "--summary"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10.0,
            check=False,
            env=vulkan_env,
        )
        output = result.stdout or ""
        has_hardware_gpu = (
            "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" in output
            or "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU" in output
        )
        if result.returncode == 0 and has_hardware_gpu:
            return

        diagnostics.append(
            f"attempt={attempt}, returncode={result.returncode}, "
            f"output_tail={output[-2000:].strip()!r}"
        )
        if attempt < 3:
            time.sleep(1.0)

    raise RuntimeError(
        "CloudRendering is selected, but Vulkan did not expose a hardware GPU "
        "after 3 attempts. Unity will likely crash with returncode=-11. "
        "Install or expose the NVIDIA Vulkan ICD (for example nvidia-driver/libnvidia-gl "
        "matching the host driver), then confirm `vulkaninfo --summary` lists an NVIDIA GPU "
        "before rerunning. "
        f"Diagnostic: executable={vulkaninfo!r}, "
        f"VK_DRIVER_FILES={vulkan_env.get('VK_DRIVER_FILES')!r}, "
        f"VK_ICD_FILENAMES={vulkan_env.get('VK_ICD_FILENAMES')!r}, "
        f"LD_LIBRARY_PATH={vulkan_env.get('LD_LIBRARY_PATH')!r}, "
        f"attempts={diagnostics!r}."
    )


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def json_safe(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:
        np = None

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if np is not None and isinstance(value, np.generic):
        return value.item()
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def compact_visible_objects(metadata: dict[str, Any], limit: int = 20) -> list[dict[str, Any]]:
    visible = []
    for obj in metadata.get("objects") or []:
        if not isinstance(obj, dict) or not obj.get("visible"):
            continue
        visible.append(
            {
                "objectId": obj.get("objectId"),
                "objectType": obj.get("objectType"),
                "name": obj.get("name"),
                "distance": obj.get("distance"),
                "position": obj.get("position"),
                "pickupable": obj.get("pickupable"),
                "openable": obj.get("openable"),
                "isOpen": obj.get("isOpen"),
            }
        )
        if len(visible) >= limit:
            break
    return visible


def compact_agent_state(agent_id: int, metadata: dict[str, Any], *, full_metadata: bool) -> dict[str, Any]:
    state = {
        "agent_id": str(agent_id),
        "agent": metadata.get("agent"),
        "lastAction": metadata.get("lastAction"),
        "lastActionSuccess": metadata.get("lastActionSuccess"),
        "errorMessage": metadata.get("errorMessage"),
        "inventoryObjects": metadata.get("inventoryObjects") or metadata.get("inventory") or [],
        "visible_objects": compact_visible_objects(metadata),
    }
    if full_metadata:
        state["metadata"] = metadata
    return json_safe(state)


def all_agent_events(event: Any) -> list[Any]:
    events = getattr(event, "events", None)
    if events is not None:
        return list(events)
    return [event]


def collect_agent_states(event: Any, *, full_metadata: bool) -> list[dict[str, Any]]:
    return [
        compact_agent_state(agent_id, dict(agent_event.metadata or {}), full_metadata=full_metadata)
        for agent_id, agent_event in enumerate(all_agent_events(event))
    ]


def depth_to_uint8(depth_frame: Any) -> Any:
    import numpy as np

    depth = np.asarray(depth_frame, dtype=np.float32)
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        return np.zeros(depth.shape, dtype=np.uint8)
    max_depth = max(float(np.percentile(valid, 95)), 1e-6)
    normalized = np.clip(depth / max_depth, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def make_agent_video_frame(event: Any) -> Any | None:
    import numpy as np

    frames = []
    for agent_event in all_agent_events(event):
        frame = getattr(agent_event, "frame", None)
        if frame is None:
            continue
        arr = np.asarray(frame).copy()
        if arr.ndim != 3:
            continue
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        frames.append(arr)

    if not frames:
        return None

    max_height = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] == max_height:
            padded.append(frame)
            continue
        pad_height = max_height - frame.shape[0]
        padded.append(np.pad(frame, ((0, pad_height), (0, 0), (0, 0)), mode="constant"))
    return np.concatenate(padded, axis=1)


class MP4VideoWriter:
    """Small PyAV-backed RGB writer with the append_data API used by recorders."""

    def __init__(self, path: Path, fps: float):
        import av
        from fractions import Fraction

        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.container = av.open(str(path), mode="w")
        self.stream = self.container.add_stream("libx264", rate=Fraction(str(fps)))
        self.initialized = False
        self.size: tuple[int, int] | None = None

    def append_data(self, frame: Any) -> None:
        import av
        import numpy as np

        array = np.asarray(frame, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] not in {3, 4}:
            raise ValueError(f"expected HxWx3 RGB frame, got shape {array.shape}")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        pad_height = array.shape[0] % 2
        pad_width = array.shape[1] % 2
        if pad_height or pad_width:
            array = np.pad(array, ((0, pad_height), (0, pad_width), (0, 0)), mode="edge")
        frame_size = (array.shape[1], array.shape[0])
        if not self.initialized:
            self.stream.width, self.stream.height = frame_size
            self.stream.pix_fmt = "yuv420p"
            self.size = frame_size
            self.initialized = True
        elif frame_size != self.size:
            raise ValueError(f"video frame size changed from {self.size} to {frame_size}")
        video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(array), format="rgb24")
        for packet in self.stream.encode(video_frame):
            self.container.mux(packet)

    def close(self) -> None:
        if self.container is None:
            return
        if self.initialized:
            for packet in self.stream.encode():
                self.container.mux(packet)
        self.container.close()
        self.container = None


class SampleVideoRecorder:
    def __init__(self, sample_dir: Path, args: argparse.Namespace):
        self.enabled = bool(args.save_agent_video)
        self.path = sample_dir / "videos" / "agent_operations.mp4"
        self.fps = args.fps
        self.writer = None
        self.frame_count = 0

    def capture(self, event: Any, _metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        frame = make_agent_video_frame(event)
        if frame is None:
            return
        if self.writer is None:
            self.writer = MP4VideoWriter(self.path, self.fps)
        self.writer.append_data(frame)
        self.frame_count += 1

    def close(self) -> str | None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.enabled and self.frame_count > 0:
            return str(self.path)
        return None


def make_controller(args: argparse.Namespace) -> Controller:
    if not args.commit_id and not args.branch and not args.local_build and not args.local_executable_path:
        if shutil.which("git") is None:
            raise RuntimeError(
                "AI2-THOR needs git to infer the local Unity build commit. Install git, or pass "
                "--commit-id, --branch, --local-build, or --local-executable-path."
            )
    scene = get_scene(args.scene_name)
    kwargs: dict[str, Any] = {
        "scene": scene,
        "agentCount": args.agentnum,
        "width": args.width,
        "height": args.height,
        "gridSize": args.grid_size,
        "visibilityDistance": args.visibility_distance,
        "fieldOfView": args.field_of_view,
        "quality": args.quality,
        "renderDepthImage": True,
        "renderInstanceSegmentation": True,
        "server_timeout": args.server_timeout,
        "server_start_timeout": args.server_start_timeout,
    }
    if args.x_display:
        kwargs["x_display"] = args.x_display
    if args.gpu_device is not None:
        kwargs["gpu_device"] = args.gpu_device
    if args.commit_id:
        kwargs["commit_id"] = args.commit_id
    if args.branch:
        kwargs["branch"] = args.branch
    if args.local_build:
        kwargs["local_build"] = True
    if args.local_executable_path:
        kwargs["local_executable_path"] = args.local_executable_path
    if args.platform == "cloud":
        kwargs["platform"] = CloudRendering
    return Controller(**kwargs)


def reset_scene(controller: Controller, args: argparse.Namespace) -> None:
    scene = get_scene(args.scene_name)
    controller.reset(
        scene,
        agentCount=args.agentnum,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
        gridSize=args.grid_size,
        visibilityDistance=args.visibility_distance,
        fieldOfView=args.field_of_view,
    )


def save_observation_dataset(
    controller: Controller,
    event: Any,
    dataset_root: Path,
    scene_id: str,
    args: argparse.Namespace,
) -> Path:
    import imageio.v2 as imageio
    import numpy as np
    from conceptgraph.utils.ai2thor import compute_intrinsics, get_camera_pose_from_event

    scene_dir = dataset_root / scene_id
    for subdir in ("color", "depth", "pose", "instance"):
        (scene_dir / subdir).mkdir(parents=True, exist_ok=True)

    k_matrix = compute_intrinsics(args.field_of_view, args.height, args.width)
    np.savetxt(scene_dir / "intrinsics.txt", k_matrix)

    frames = all_agent_events(event)
    for frame_idx, agent_event in enumerate(frames):
        color = np.asarray(agent_event.frame).copy()
        depth_frame = getattr(agent_event, "depth_frame", None)
        if depth_frame is None:
            depth = np.zeros((args.height, args.width), dtype=np.float32)
        else:
            depth = np.asarray(depth_frame, dtype=np.float32).copy()
        depth[depth > 15] = 0
        depth_png = np.round(depth * 1000.0).astype(np.uint16)
        try:
            camera_pose = get_camera_pose_from_event(agent_event)
        except Exception:
            camera_pose = np.eye(4, dtype=np.float32)

        imageio.imwrite(scene_dir / "color" / f"{frame_idx:06d}.png", color)
        imageio.imwrite(scene_dir / "depth" / f"{frame_idx:06d}.png", depth_png)
        np.savetxt(scene_dir / "pose" / f"{frame_idx:06d}.txt", camera_pose)

        instance_frame = getattr(agent_event, "instance_segmentation_frame", None)
        if instance_frame is not None:
            imageio.imwrite(scene_dir / "instance" / f"{frame_idx:06d}.png", np.asarray(instance_frame).copy())
        else:
            imageio.imwrite(scene_dir / "instance" / f"{frame_idx:06d}.png", np.zeros_like(color))

    first_event = frames[0] if frames else event
    save_json(json_safe(first_event.metadata.get("objects", [])), scene_dir / "obj_meta.json")
    return scene_dir


def run_scenegraph_pipeline(dataset_root: Path, scene_id: str, cachedir: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.disable_scenegraph:
        cachedir.mkdir(parents=True, exist_ok=True)
        placeholder = {
            "disabled": True,
            "reason": "--disable-scenegraph was set",
            "dataset_root": str(dataset_root),
            "scene_id": scene_id,
        }
        save_json(placeholder, cachedir / "scene_graph_placeholder.json")
        return {
            "cachedir": str(cachedir),
            "scene_graph": None,
            "relations": None,
            "placeholder": str(cachedir / "scene_graph_placeholder.json"),
        }

    from conceptgraph.scripts import run_full_scenegraph_pipeline

    pipeline_args = run_full_scenegraph_pipeline.build_pipeline_args(
        dataset_root=dataset_root,
        dataset_config=Path(args.dataset_config),
        scene_id=scene_id,
        start=0,
        end=-1,
        stride=args.scenegraph_stride,
        desired_height=args.height,
        desired_width=args.width,
        device=args.scenegraph_device,
        scenegraph_device=args.scenegraph_build_device,
        obj_min_detections=args.scenegraph_obj_min_detections,
        min_views_per_object=args.scenegraph_min_views_per_object,
        class_set=args.class_set,
        cachedir=cachedir,
        vlm_backend=args.vlm_backend,
        vlm_model_path=Path(args.qwen_model_path),
        vlm_conv_mode=args.qwen_conv_mode,
        vlm_num_gpus=args.qwen_num_gpus,
        generate_scenegraph_json=True,
        skip_existing=args.scenegraph_skip_existing,
        dry_run=args.scenegraph_dry_run,
    )

    log_memory(f"before scenegraph {scene_id}", args)
    log(f"running ConceptGraphs pipeline for {scene_id}")
    outputs = run_full_scenegraph_pipeline.run_pipeline(pipeline_args)
    log_memory(f"after scenegraph {scene_id}", args)

    return {
        "cachedir": str(outputs["cachedir"]),
        "scene_graph": str(outputs["scene_graph"]) if outputs["scene_graph"] is not None else None,
        "relations": str(outputs["relations"]),
        "object_relations": str(outputs["object_relations"]),
    }


def build_subgraph_and_task_graph(scenegraph_info: dict[str, Any], task: str, output_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    subgraph_path = output_dir / "task_relevant_subgraph.json"
    task_graph_path = output_dir / "task_graph.json"

    subgraph = extract_task_relevant_subgraph_to_file(
        scenegraph_info=scenegraph_info,
        task=task,
        output_path=subgraph_path,
        args=args,
    )

    task_graph = decompose_task_to_graph(
        task=task,
        subgraph=subgraph,
        use_qwen=not args.disable_qwen,
        qwen_model_path=args.planning_model_path,
        qwen_conv_mode=args.qwen_conv_mode,
        qwen_num_gpus=args.qwen_num_gpus,
        planning_max_attempts=max(int(getattr(args, "planning_max_attempts", 3)), 1),
    )
    save_task_graph(task_graph, task_graph_path)
    return {
        "subgraph_path": str(subgraph_path),
        "task_graph_path": str(task_graph_path),
        "subgraph": subgraph,
        "task_graph": task_graph,
    }


def task_index(task_graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(task["id"]): deepcopy(task) for task in task_graph.get("flat_tasks", [])}


def empty_task_subgraph(task: str, parser: str, reason: str) -> dict[str, Any]:
    return {
        "mode": "task",
        "task": task,
        "task_spec": {"raw_task": task, "parser": parser, "reason": reason},
        "seed_nodes": [],
        "nodes": [],
        "edges": [],
        "triples": [],
        "candidate_paths": [],
    }


def extract_task_relevant_subgraph_to_file(
    scenegraph_info: dict[str, Any],
    task: str,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if args.disable_scenegraph:
        subgraph = empty_task_subgraph(task, "disabled_scenegraph", "--disable-scenegraph was set")
        save_json(subgraph, output_path)
        return subgraph

    scene_graph_path_raw = scenegraph_info.get("scene_graph")
    relations_path_raw = scenegraph_info.get("relations")
    if not scene_graph_path_raw or not relations_path_raw:
        subgraph = empty_task_subgraph(task, "missing_scenegraph", "scene graph or relations path missing")
        save_json(subgraph, output_path)
        return subgraph

    scene_graph_path = Path(scene_graph_path_raw)
    relations_path = Path(relations_path_raw)
    if not scene_graph_path.exists() or not relations_path.exists():
        subgraph = empty_task_subgraph(
            task,
            "missing_scenegraph_file",
            f"missing input file: scene_graph={scene_graph_path.exists()} relations={relations_path.exists()}",
        )
        save_json(subgraph, output_path)
        return subgraph

    return extract_subgraph(
        scene_graph=scene_graph_path,
        relations=relations_path,
        task=task,
        output=output_path,
    )


def summarize_tasks_for_subgraph(tasks: list[dict[str, Any]], overall_task: str) -> str:
    lines = [f"Overall task: {overall_task}", "Current executable subtasks:"]
    for index, task in enumerate(tasks, start=1):
        grounding = task.get("grounding") or {}
        object_tags = ", ".join(str(item) for item in grounding.get("object_tags") or [])
        relation_texts = ", ".join(str(item) for item in grounding.get("relation_texts") or [])
        parts = [
            f"id={task.get('id')}",
            f"name={task.get('name') or task.get('action') or f'task_{index}'}",
        ]
        if task.get("description"):
            parts.append(f"description={task['description']}")
        if object_tags:
            parts.append(f"objects={object_tags}")
        if relation_texts:
            parts.append(f"relations={relation_texts}")
        lines.append(f"{index}. " + "; ".join(parts))
    return "\n".join(lines)


def build_loop_task_subgraph(
    scenegraph_info: dict[str, Any],
    tasks: list[dict[str, Any]],
    overall_task: str,
    output_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    task_query = summarize_tasks_for_subgraph(tasks, overall_task)
    subgraph = extract_task_relevant_subgraph_to_file(
        scenegraph_info=scenegraph_info,
        task=task_query,
        output_path=output_path,
        args=args,
    )
    return {
        "task_ids": [str(task.get("id")) for task in tasks],
        "task_query": task_query,
        "source_scene_graph": scenegraph_info.get("scene_graph"),
        "source_relations": scenegraph_info.get("relations"),
        "subgraph_path": str(output_path),
        "num_nodes": len(subgraph.get("nodes") or []),
        "num_edges": len(subgraph.get("edges") or []),
        "subgraph": subgraph,
    }


def remaining_tasks(
    task_graph: dict[str, Any],
    completed: set[str],
    failed: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Return every unfinished semantic task in stable Task Graph order."""

    terminal = set(completed) | set(failed or set())
    return [
        deepcopy(task)
        for task in task_graph.get("flat_tasks", [])
        if str(task.get("id")) not in terminal
    ]


def runtime_replan_source_key(
    *,
    task_graph_version: int,
    trigger: str,
    evidence: dict[str, Any],
) -> str:
    payload = {
        "task_graph_version": int(task_graph_version),
        "trigger": str(trigger),
        "evidence": evidence,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"v{task_graph_version}:{trigger}:{digest}"


def replan_datagen_task_graph(
    *,
    root_task: str,
    current_task_graph: dict[str, Any],
    scene_context: dict[str, Any],
    agent_states: list[dict[str, Any]],
    completed: set[str],
    failed: set[str],
    trigger: str,
    trigger_evidence: dict[str, Any],
    task_graph_version: int,
    replan_index: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Replace the semantic graph from current runtime facts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    context = {
        "mode": "runtime_replanning",
        "trigger": trigger,
        "root_task": root_task,
        "task_graph_version": int(task_graph_version),
        "completed_task_ids": sorted(completed),
        "failed_task_ids": sorted(failed),
        "remaining_tasks_before_replan": remaining_tasks(
            current_task_graph, completed, failed
        ),
        "agent_states": deepcopy(agent_states),
        "trigger_evidence": deepcopy(trigger_evidence),
    }
    save_json(current_task_graph, output_dir / "old_task_graph.json")
    save_json(context, output_dir / "runtime_context.json")
    try:
        replanned = decompose_task_to_graph(
            task=root_task,
            subgraph=scene_context,
            use_qwen=not bool(getattr(args, "disable_qwen", False)),
            qwen_model_path=args.planning_model_path,
            qwen_conv_mode=args.qwen_conv_mode,
            qwen_num_gpus=args.qwen_num_gpus,
            qwen_max_new_tokens=args.planning_max_new_tokens,
            agent_context={
                "agent_count": len(agent_states)
                or max(int(getattr(args, "agentnum", 1)), 1),
                "agents": deepcopy(agent_states),
                "inventory_capacity_per_agent": 1,
            },
            planning_max_attempts=max(
                int(getattr(args, "task_graph_replan_max_attempts", 3)), 1
            ),
            planning_mode="runtime_replan",
            execution_context=context,
        )
        replanned = rebuild_task_graph_views(replanned)
        replanned["task_graph_version"] = int(task_graph_version) + 1
        replanned.setdefault("runtime_history", {})["replanning"] = {
            "replan_index": int(replan_index),
            "trigger": trigger,
            "previous_task_graph": str(output_dir / "old_task_graph.json"),
            "runtime_context": str(output_dir / "runtime_context.json"),
        }
        save_task_graph(replanned, output_dir / "new_task_graph.json")
        return {
            "status": "success",
            "trigger": trigger,
            "replan_index": int(replan_index),
            "task_graph": replanned,
            "task_graph_path": str(output_dir / "new_task_graph.json"),
        }
    except Exception as exc:
        diagnostics = (
            exc.to_dict()
            if isinstance(exc, TaskPlanningError)
            else {"error_type": type(exc).__name__, "error": str(exc)}
        )
        result = {
            "status": "failed",
            "trigger": trigger,
            "replan_index": int(replan_index),
            "reason": "planning_failed",
            "diagnostics": diagnostics,
        }
        save_json(result, output_dir / "replan_failure.json")
        return result


def agent_state_index(agent_states: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("agent_id")): deepcopy(item) for item in agent_states}


def trace_index(traces: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("subtask_id")): deepcopy(item) for item in traces}


def evaluate_task_statuses(
    assignments: list[dict[str, Any]],
    execution: dict[str, Any],
    pre_agent_states: list[dict[str, Any]],
    post_agent_states: list[dict[str, Any]],
    retry_counts: dict[str, int],
    max_task_retries: int,
) -> list[dict[str, Any]]:
    traces_by_task = trace_index(execution.get("traces") or [])
    pre_agents = agent_state_index(pre_agent_states)
    post_agents = agent_state_index(post_agent_states)
    statuses = []

    for assignment in assignments:
        subtask = deepcopy(assignment.get("subtask") or {})
        task_id = str(subtask.get("id"))
        agent_id = str(assignment.get("agent_id", 0))
        trace = traces_by_task.get(task_id, {})
        status = judge_task_status(
            subtask=subtask,
            trace=trace,
            pre_agent_state=pre_agents.get(agent_id, {}),
            post_agent_state=post_agents.get(agent_id, {}),
            retry_count=retry_counts.get(task_id, 0),
            max_retry=max_task_retries,
        )
        status["execution_time_seconds"] = float(trace.get("execution_time_seconds") or 0.0)
        status["simulation_time_seconds"] = trace.get("simulation_time_seconds")
        status["collided"] = bool(trace.get("collided"))
        status["collision_count"] = int(trace.get("collision_count") or 0)
        status["collided_objects"] = list(trace.get("collided_objects") or [])
        status["agent_id"] = agent_id
        status["subtask"] = subtask
        statuses.append(status)

    return statuses


def scenegraph_node_key(node: dict[str, Any]) -> tuple[str, ...]:
    object_id = node.get("original_id", node.get("id"))
    if object_id not in (None, ""):
        return ("object_id", str(object_id))
    return (
        "object_text",
        str(node.get("object_tag", "")).strip().lower(),
        str(node.get("caption", "")).strip().lower(),
    )


MUTATING_SUBTASK_ACTIONS = {
    "pick",
    "place",
    "open",
    "close",
    "toggle_on",
    "toggle_off",
    "turn_on",
    "turn_off",
    "switch_on",
    "switch_off",
    "slice",
    "break",
    "dirty",
    "clean",
    "fill",
    "empty",
    "use_up",
    "cook",
}

SUPPORTED_RELATION_RECORDS = {"a on b", "b on a", "a in b", "b in a"}


def canonical_object_id(value: Any) -> int | str | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def node_canonical_object_id(node: dict[str, Any]) -> int | str | None:
    return canonical_object_id(node.get("original_id", node.get("id", node.get("pruned_id"))))


def load_scenegraph_nodes(scenegraph_path: Path | None) -> list[dict[str, Any]]:
    if scenegraph_path is None or not scenegraph_path.exists():
        return []
    nodes = load_json(scenegraph_path)
    if not isinstance(nodes, list):
        return []
    return [deepcopy(node) for node in nodes if isinstance(node, dict)]


def canonicalize_scenegraph_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_nodes = []
    for pruned_id, node in enumerate(nodes):
        canonical_node = deepcopy(node)
        canonical_node["pruned_id"] = pruned_id
        canonical_node.setdefault("original_id", canonical_node.get("id", pruned_id))
        canonical_nodes.append(canonical_node)
    return canonical_nodes


def load_relation_records(relations_path: Path | None, scenegraph_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if relations_path is None or not relations_path.exists():
        return []

    node_id_to_original: dict[int | str, int | str] = {}
    for node in scenegraph_nodes:
        original_id = node_canonical_object_id(node)
        if original_id is None:
            continue
        for key in ("pruned_id", "original_id", "id"):
            node_id = canonical_object_id(node.get(key))
            if node_id is not None:
                node_id_to_original[node_id] = original_id

    if relations_path.suffix == ".json":
        payload = load_json(relations_path)
        if not isinstance(payload, list):
            return []
        records = []
        for relation in payload:
            if not isinstance(relation, dict):
                continue
            relation_name = str(relation.get("object_relation", "")).strip().lower()
            if relation_name not in SUPPORTED_RELATION_RECORDS:
                continue
            object1 = deepcopy(relation.get("object1") or {})
            object2 = deepcopy(relation.get("object2") or {})
            object1_id = canonical_object_id(object1.get("id"))
            object2_id = canonical_object_id(object2.get("id"))
            if object1_id is None or object2_id is None:
                continue
            object1["id"] = node_id_to_original.get(object1_id, object1_id)
            object2["id"] = node_id_to_original.get(object2_id, object2_id)
            records.append({"object1": object1, "object2": object2, "object_relation": relation_name})
        return records

    if relations_path.suffix == ".pkl":
        with open(relations_path, "rb") as f:
            raw_edges = pickle.load(f)
        records = []
        for item in raw_edges:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            object1_id_raw, object2_id_raw, relation_name_raw = item
            relation_name = str(relation_name_raw).strip().lower()
            if relation_name not in SUPPORTED_RELATION_RECORDS:
                continue
            object1_id = canonical_object_id(object1_id_raw)
            object2_id = canonical_object_id(object2_id_raw)
            if object1_id is None or object2_id is None:
                continue
            records.append(
                {
                    "object1": {"id": node_id_to_original.get(object1_id, object1_id)},
                    "object2": {"id": node_id_to_original.get(object2_id, object2_id)},
                    "object_relation": relation_name,
                }
            )
        return records

    return []


def relation_record_key(relation: dict[str, Any]) -> tuple[str, str, str] | None:
    relation_name = str(relation.get("object_relation", "")).strip().lower()
    object1_id = canonical_object_id((relation.get("object1") or {}).get("id"))
    object2_id = canonical_object_id((relation.get("object2") or {}).get("id"))
    if relation_name not in SUPPORTED_RELATION_RECORDS or object1_id is None or object2_id is None:
        return None
    return (str(object1_id), str(object2_id), relation_name)


def relation_record_endpoint_ids(relation: dict[str, Any]) -> tuple[int | str | None, int | str | None]:
    object1_id = canonical_object_id((relation.get("object1") or {}).get("id"))
    object2_id = canonical_object_id((relation.get("object2") or {}).get("id"))
    return object1_id, object2_id


def subgraph_nodes_by_pruned_id(subgraph: dict[str, Any]) -> dict[int, dict[str, Any]]:
    nodes = {}
    for node in subgraph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        try:
            pruned_id = int(node.get("pruned_id"))
        except (TypeError, ValueError):
            continue
        nodes[pruned_id] = node
    return nodes


def relation_record_from_subgraph_edge(edge: dict[str, Any], subgraph_nodes: dict[int, dict[str, Any]]) -> dict[str, Any] | None:
    relation_name = str(edge.get("object_relation", "")).strip().lower()
    if relation_name not in SUPPORTED_RELATION_RECORDS:
        return None

    try:
        source_pruned_id = int(edge.get("source"))
        target_pruned_id = int(edge.get("target"))
    except (TypeError, ValueError):
        return None

    source_node = subgraph_nodes.get(source_pruned_id)
    target_node = subgraph_nodes.get(target_pruned_id)
    if source_node is None or target_node is None:
        return None

    source_id = node_canonical_object_id(source_node)
    target_id = node_canonical_object_id(target_node)
    if source_id is None or target_id is None:
        return None

    if relation_name in {"a on b", "a in b"}:
        object1_id = target_id
        object2_id = source_id
    else:
        object1_id = source_id
        object2_id = target_id

    return {
        "object1": {"id": object1_id},
        "object2": {"id": object2_id},
        "object_relation": relation_name,
    }


def relation_records_from_subgraph(subgraph: dict[str, Any]) -> list[dict[str, Any]]:
    nodes_by_pruned_id = subgraph_nodes_by_pruned_id(subgraph)
    records = []
    seen = set()
    for edge in subgraph.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        record = relation_record_from_subgraph_edge(edge, nodes_by_pruned_id)
        if record is None:
            continue
        key = relation_record_key(record)
        if key is None or key in seen:
            continue
        seen.add(key)
        records.append(record)
    return records


def focus_object_ids_from_subgraph(subgraph: dict[str, Any]) -> set[int | str]:
    focus_ids = set()
    for group_name in ("seed_nodes", "nodes"):
        for node in subgraph.get(group_name) or []:
            if not isinstance(node, dict):
                continue
            object_id = node_canonical_object_id(node)
            if object_id is not None:
                focus_ids.add(object_id)
        if focus_ids:
            break
    return focus_ids


def assignments_modify_scene(assignments: list[dict[str, Any]]) -> bool:
    for assignment in assignments:
        subtask = assignment.get("subtask") or {}
        action = str(subtask.get("action", "")).strip().lower()
        if action in MUTATING_SUBTASK_ACTIONS:
            return True
    return False


def merge_scenegraph_with_task_subgraphs(
    current_scenegraph: dict[str, Any],
    pre_task_subgraph: dict[str, Any],
    post_task_subgraph: dict[str, Any],
    assignments: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    merged_scenegraph_path = output_dir / "scene_graph.json"
    merged_relations_path = output_dir / "cfslam_object_relations.json"
    merge_metadata_path = output_dir / "merge_metadata.json"

    current_scenegraph_path = Path(current_scenegraph["scene_graph"]) if current_scenegraph.get("scene_graph") else None
    current_relations_raw = current_scenegraph.get("relations") or current_scenegraph.get("object_relations")
    current_relations_path = Path(current_relations_raw) if current_relations_raw else None
    current_nodes = load_scenegraph_nodes(current_scenegraph_path)
    current_relations = load_relation_records(current_relations_path, current_nodes)

    pre_subgraph = deepcopy(pre_task_subgraph.get("subgraph") or {})
    post_subgraph = deepcopy(post_task_subgraph.get("subgraph") or {})
    post_nodes = [deepcopy(node) for node in post_subgraph.get("nodes") or [] if isinstance(node, dict)]
    post_relation_records = relation_records_from_subgraph(post_subgraph)

    nodes_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    ordered_keys: list[tuple[str, ...]] = []
    replaced_keys = 0
    for source, node_list in (("current", current_nodes), ("post_task_subgraph", post_nodes)):
        for node in node_list:
            key = scenegraph_node_key(node)
            merged_node = deepcopy(node)
            merged_node["merge_source"] = source
            if key in nodes_by_key and source == "post_task_subgraph":
                replaced_keys += 1
            elif key not in nodes_by_key:
                ordered_keys.append(key)
            nodes_by_key[key] = merged_node
    merged_nodes = canonicalize_scenegraph_nodes([nodes_by_key[key] for key in ordered_keys])

    active_object_ids = focus_object_ids_from_subgraph(pre_subgraph) | focus_object_ids_from_subgraph(post_subgraph)
    task_mutates_scene = assignments_modify_scene(assignments)

    kept_relations = []
    removed_relation_keys = set()
    for relation in current_relations:
        key = relation_record_key(relation)
        if key is None:
            continue
        object1_id, object2_id = relation_record_endpoint_ids(relation)
        touches_active_object = object1_id in active_object_ids or object2_id in active_object_ids
        if task_mutates_scene and touches_active_object:
            removed_relation_keys.add(key)
            continue
        kept_relations.append(relation)

    relation_index = {relation_record_key(relation): deepcopy(relation) for relation in kept_relations if relation_record_key(relation) is not None}
    added_or_updated_relations = 0
    for relation in post_relation_records:
        key = relation_record_key(relation)
        if key is None:
            continue
        object1_id, object2_id = relation_record_endpoint_ids(relation)
        touches_active_object = object1_id in active_object_ids or object2_id in active_object_ids
        if active_object_ids and not touches_active_object:
            continue
        if key not in relation_index:
            added_or_updated_relations += 1
        relation_index[key] = deepcopy(relation)

    merged_relations = list(relation_index.values())

    save_json(merged_nodes, merged_scenegraph_path)
    save_json(merged_relations, merged_relations_path)

    merge_metadata = {
        "merge_backend": "task_subgraph_replace_merge",
        "scene_graph": str(merged_scenegraph_path),
        "relations": str(merged_relations_path),
        "source_scenegraph": current_scenegraph.get("scene_graph"),
        "source_relations": current_relations_raw,
        "pre_task_subgraph": pre_task_subgraph.get("subgraph_path"),
        "post_task_subgraph": post_task_subgraph.get("subgraph_path"),
        "num_nodes_before": len(current_nodes),
        "num_nodes_after": len(merged_nodes),
        "num_node_updates": replaced_keys,
        "num_relations_before": len(current_relations),
        "num_relations_after": len(merged_relations),
        "num_relations_removed": len(removed_relation_keys),
        "num_relations_added_or_updated": added_or_updated_relations,
        "active_object_ids": [str(object_id) for object_id in sorted(active_object_ids, key=str)],
        "task_mutates_scene": task_mutates_scene,
        "assignment_actions": [str((assignment.get("subtask") or {}).get("action", "")) for assignment in assignments],
    }
    save_json(merge_metadata, merge_metadata_path)

    return {
        "cachedir": str(output_dir),
        "scene_graph": str(merged_scenegraph_path),
        "relations": str(merged_relations_path),
        "object_relations": str(merged_relations_path),
        "merge_backend": merge_metadata["merge_backend"],
        "merge_metadata": str(merge_metadata_path),
        "initial_scenegraph": current_scenegraph.get("scene_graph"),
        "initial_relations": current_relations_raw,
    }


def merge_scenegraph_info(
    current_scenegraph: dict[str, Any],
    pre_task_subgraph: dict[str, Any],
    post_task_subgraph: dict[str, Any],
    assignments: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    return merge_scenegraph_with_task_subgraphs(
        current_scenegraph=current_scenegraph,
        pre_task_subgraph=pre_task_subgraph,
        post_task_subgraph=post_task_subgraph,
        assignments=assignments,
        output_dir=output_dir,
    )


def collect_stage(
    controller: Controller,
    sample_dir: Path,
    stage_name: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    event = controller.last_event
    stage_dir = sample_dir / "stages" / stage_name
    dataset_root = stage_dir / "conceptgraph_dataset"
    scene_id = stage_name
    scene_dir = save_observation_dataset(controller, event, dataset_root, scene_id, args)
    agent_states = collect_agent_states(event, full_metadata=args.full_metadata)
    instance_color_maps: dict[str, dict[str, str]] = {}
    observed_object_ids: set[str] = set()
    for frame_idx, agent_event in enumerate(all_agent_events(event)):
        color_map = getattr(agent_event, "color_to_object_id", {}) or {}
        serialized = {
            ",".join(str(int(channel)) for channel in color): str(object_id)
            for color, object_id in color_map.items()
        }
        instance_color_maps[f"{frame_idx:06d}.png"] = serialized
        for obj in (agent_event.metadata or {}).get("objects") or []:
            if isinstance(obj, dict) and obj.get("visible") and obj.get("objectId"):
                observed_object_ids.add(str(obj["objectId"]))
    save_json(agent_states, stage_dir / "agent_states.json")
    return {
        "stage_name": stage_name,
        "stage_dir": str(stage_dir),
        "dataset_root": str(dataset_root),
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "agent_states_path": str(stage_dir / "agent_states.json"),
        "agent_states": agent_states,
        "instance_color_maps": instance_color_maps,
        "observed_object_ids": sorted(observed_object_ids),
    }


def build_stage_scenegraph(stage_info: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cachedir = Path(stage_info["stage_dir"]) / "sg_cache"
    return run_scenegraph_pipeline(
        dataset_root=Path(stage_info["dataset_root"]),
        scene_id=stage_info["scene_id"],
        cachedir=cachedir,
        args=args,
    )


def update_stage_scenegraph_incrementally(
    *,
    pre_stage: dict[str, Any],
    post_stage: dict[str, Any],
    current_scenegraph: dict[str, Any],
    execution: dict[str, Any],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Detect unseen post-stage objects and append them to the current graph."""

    stored_scenegraph_raw = current_scenegraph.get("scene_graph")
    stored_relations_raw = current_scenegraph.get("object_relations") or current_scenegraph.get("relations")
    if not stored_scenegraph_raw or not stored_relations_raw:
        observed = build_stage_scenegraph(post_stage, args)
        return observed, observed

    stored_scenegraph_path = Path(stored_scenegraph_raw)
    stored_relations_path = Path(stored_relations_raw)
    observed = build_incremental_observed_scenegraph(
        dataset_root=Path(post_stage["dataset_root"]),
        scene_id=str(post_stage["scene_id"]),
        cachedir=Path(post_stage["stage_dir"]) / "incremental_sg_cache",
        stored_scenegraph_path=stored_scenegraph_path,
        args=args,
        instance_color_maps=post_stage.get("instance_color_maps"),
        observed_object_ids=set(post_stage.get("observed_object_ids") or []),
    )

    pre_metadata = load_json(Path(pre_stage["scene_dir"]) / "obj_meta.json")
    post_metadata = load_json(Path(post_stage["scene_dir"]) / "obj_meta.json")
    execution_deltas = infer_relation_deltas_from_metadata(
        pre_metadata,
        post_metadata,
        execution=execution,
    )
    relation_deltas: list[Any] = [delta.to_dict() for delta in execution_deltas]
    object_deltas = [delta.to_dict() for delta in diff_objects(pre_metadata, post_metadata)]
    relation_deltas.extend(observed.get("relation_deltas") or [])
    output_dir.mkdir(parents=True, exist_ok=True)
    relation_delta_path = output_dir / "relation_deltas.json"
    object_delta_path = output_dir / "object_deltas.json"
    save_json(
        {
            "execution_deltas": [delta.to_dict() for delta in execution_deltas],
            "new_object_deltas": observed.get("relation_deltas") or [],
            "all_deltas": relation_deltas,
        },
        relation_delta_path,
    )
    save_json(object_deltas, object_delta_path)

    observed_scenegraph_path = None
    if observed.get("scene_graph"):
        observed_scenegraph_path = Path(observed["scene_graph"])
    updated = update_scenegraph_files(
        stored_scenegraph_path=stored_scenegraph_path,
        stored_relations_path=stored_relations_path,
        observed_scenegraph_path=observed_scenegraph_path,
        relation_deltas=relation_deltas,
        object_deltas=object_deltas,
        output_dir=output_dir,
        append_observed_only=True,
    )
    updated["update_backend"] = "incremental_observation"
    updated["incremental_observation"] = observed
    updated["relation_deltas"] = str(relation_delta_path)
    updated["object_deltas"] = str(object_delta_path)
    return observed, updated


def collect_global_mapping_stage(
    controller: Controller,
    sample_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from conceptgraph.scripts.generate_ai2thor_dataset import generate_obs_from_poses
    from conceptgraph.utils.ai2thor import compute_intrinsics, sample_pose_random, sample_pose_uniform

    stage_name = "global_mapping"
    stage_dir = sample_dir / "stages" / stage_name
    dataset_root = stage_dir / "conceptgraph_dataset"
    scene_id = stage_name
    scene_dir = dataset_root / scene_id
    stage_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    if args.global_map_sample_method == "random":
        sampled_poses = sample_pose_random(controller, args.global_map_samples)
    else:
        sampled_poses = sample_pose_uniform(controller, args.global_map_samples)

    K = compute_intrinsics(args.field_of_view, args.height, args.width)
    log(
        f"collecting global mapping observations "
        f"({len(sampled_poses)} frames, method={args.global_map_sample_method})"
    )
    generate_obs_from_poses(
        controller=controller,
        K=K,
        sampled_poses=sampled_poses,
        save_root=str(scene_dir),
        depth_scale=1000.0,
        save_video=False,
        agent_id=args.global_map_agent_id,
    )

    agent_states = collect_agent_states(controller.last_event, full_metadata=args.full_metadata)
    save_json(agent_states, stage_dir / "agent_states_after_mapping.json")
    save_json(
        {
            "stage_name": stage_name,
            "sample_method": args.global_map_sample_method,
            "global_map_samples": args.global_map_samples,
            "num_sampled_poses": len(sampled_poses),
            "note": "Generated with conceptgraph.scripts.generate_ai2thor_dataset.generate_obs_from_poses.",
        },
        stage_dir / "mapping_info.json",
    )
    return {
        "stage_name": stage_name,
        "stage_dir": str(stage_dir),
        "dataset_root": str(dataset_root),
        "scene_id": scene_id,
        "scene_dir": str(scene_dir),
        "agent_states_path": str(stage_dir / "agent_states_after_mapping.json"),
        "mapping_info_path": str(stage_dir / "mapping_info.json"),
        "agent_states": agent_states,
    }


def build_global_scenegraph(controller: Controller, sample_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    if args.disable_scenegraph or not args.global_scenegraph:
        return None

    mapping_stage = collect_global_mapping_stage(controller, sample_dir, args)
    scenegraph = build_stage_scenegraph(mapping_stage, args)
    return {
        "stage": {key: value for key, value in mapping_stage.items() if key != "agent_states"},
        "scenegraph": scenegraph,
    }


def run_sample(controller: Controller, sample_index: int, run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    sample_dir = run_dir / f"sample_{sample_index:06d}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    log(f"sample {sample_index}: reset scene")
    reset_scene(controller, args)
    video_recorder = SampleVideoRecorder(sample_dir, args)
    video_recorder.capture(controller.last_event, {"stage": "after_reset"})

    sample_record: dict[str, Any] = {
        "sample_index": sample_index,
        "scene_name": args.scene_name,
        "task": args.task,
        "loops": [],
    }

    global_scenegraph_record = build_global_scenegraph(controller, sample_dir, args)
    if global_scenegraph_record is not None:
        sample_record["global_mapping"] = global_scenegraph_record
        log(f"sample {sample_index}: reset scene after global mapping")
        reset_scene(controller, args)
        video_recorder.capture(controller.last_event, {"stage": "after_global_mapping_reset"})

    initial_stage = collect_stage(controller, sample_dir, "initial", args)
    if global_scenegraph_record is None:
        initial_scenegraph = build_stage_scenegraph(initial_stage, args)
    else:
        initial_scenegraph = {
            "skipped": True,
            "reason": "initial local scene graph skipped because global_mapping scene graph is used for planning",
            "source_scenegraph": global_scenegraph_record["scenegraph"],
        }
    planning_scenegraph = (
        global_scenegraph_record["scenegraph"]
        if global_scenegraph_record is not None
        else initial_scenegraph
    )
    initial_task = build_subgraph_and_task_graph(planning_scenegraph, args.task, sample_dir / "initial_planning", args)
    original_task_graph = initial_task["task_graph"]
    task_graph_version = max(int(original_task_graph.get("task_graph_version") or 1), 1)
    task_ids = set(task_index(original_task_graph))
    completed: set[str] = set()
    failed: set[str] = set()
    retry_counts: dict[str, int] = {}
    current_scenegraph = planning_scenegraph
    task_graph_replan_count = 0
    task_graph_replan_counts_by_source: dict[str, int] = {}
    task_graph_replan_events: list[dict[str, Any]] = []
    sample_record["initial"] = {
        "stage": {key: value for key, value in initial_stage.items() if key != "agent_states"},
        "scenegraph": initial_scenegraph,
        "planning_scenegraph_source": "global_mapping" if global_scenegraph_record is not None else "initial",
        "planning": {
            "task_relevant_subgraph": initial_task["subgraph_path"],
            "task_graph": initial_task["task_graph_path"],
        },
    }

    for loop_index in range(args.max_task_loops):
        if completed | failed >= task_ids:
            break

        pre_stage = collect_stage(controller, sample_dir, f"loop_{loop_index:03d}_pre", args)
        unfinished_tasks = remaining_tasks(original_task_graph, completed, failed)
        if not unfinished_tasks:
            break

        loop_dir = sample_dir / "loops" / f"loop_{loop_index:03d}"
        loop_dir.mkdir(parents=True, exist_ok=True)
        pre_task_subgraph = build_loop_task_subgraph(
            scenegraph_info=current_scenegraph,
            tasks=unfinished_tasks,
            overall_task=args.task,
            output_path=loop_dir / "pre_task_subgraph.json",
            args=args,
        )
        progress = {
            "completed_task_ids": sorted(completed),
            "failed_task_ids": sorted(failed),
        }
        allocation_request = {
            "task_graph": original_task_graph,
            "agent_states": pre_stage["agent_states"],
            "scene_context": pre_task_subgraph["subgraph"],
            "progress": progress,
            "task_graph_version": task_graph_version,
        }
        save_json({key: value for key, value in pre_stage.items() if key != "agent_states"}, loop_dir / "pre_stage_info.json")
        save_json(current_scenegraph, loop_dir / "pre_scenegraph_info.json")
        save_json(original_task_graph, loop_dir / "active_task_graph.json")
        save_json({key: value for key, value in pre_task_subgraph.items() if key != "subgraph"}, loop_dir / "pre_task_subgraph_info.json")
        save_json(pre_stage["agent_states"], loop_dir / "pre_agent_states.json")
        save_json(allocation_request, loop_dir / "allocation_request.json")

        loop_qwen_chat = None
        use_loop_qwen = not args.disable_qwen
        log_memory(f"loop {loop_index} before Qwen", args)
        if use_loop_qwen:
            try:
                from conceptgraph.vlm import build_vlm_chat

                loop_qwen_chat = build_vlm_chat(
                    backend="qwen",
                    model_path=args.planning_model_path,
                    conv_mode=args.qwen_conv_mode,
                    num_gpus=args.qwen_num_gpus,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"loop {loop_index}: Qwen initialization failed; "
                    "rerun with --disable-qwen to select the deterministic scheduler explicitly"
                ) from exc

        log_memory(f"loop {loop_index} planning start", args)
        execution_plan: dict[str, Any] | None = None
        assignments: list[dict[str, Any]] = []
        execution: dict[str, Any] | None = None
        try:
            execution_plan = build_execution_plan(
                task_graph=original_task_graph,
                agent_states=pre_stage["agent_states"],
                scene_context=pre_task_subgraph["subgraph"],
                progress=progress,
                task_graph_version=task_graph_version,
                use_qwen=use_loop_qwen,
                qwen_model_path=args.planning_model_path,
                qwen_conv_mode=args.qwen_conv_mode,
                qwen_num_gpus=args.qwen_num_gpus,
                qwen_chat=loop_qwen_chat,
                qwen_max_new_tokens=args.planning_max_new_tokens,
                allocation_max_attempts=max(int(args.planning_max_attempts), 1),
                diagnostics_output_dir=loop_dir / "allocation_attempts",
            )
            save_json(execution_plan, loop_dir / "execution_plan.json")
            if execution_plan.get("state") == "dispatchable":
                assignments = first_execution_unit(execution_plan, original_task_graph)
                save_json(
                    {
                        "state": "dispatchable",
                        "time_step": execution_plan["units"][0].get("time_step", 1),
                        "assignments": assignments,
                    },
                    loop_dir / "executed_allocation_unit.json",
                )

                video_recorder.capture(controller.last_event, {"stage": "before_execution", "loop_index": loop_index})
                execution = execute_allocated_skills(
                    controller,
                    assignments,
                    use_qwen=use_loop_qwen and not args.disable_skill_qwen,
                    qwen_model_path=args.planning_model_path,
                    qwen_conv_mode=args.qwen_conv_mode,
                    qwen_num_gpus=args.qwen_num_gpus,
                    max_steps=args.skill_max_steps,
                    complete_on_execute=args.skill_complete_on_execute,
                    step_callback=video_recorder.capture,
                    qwen_chat=loop_qwen_chat,
                )
            else:
                save_json(
                    {
                        "state": "blocked",
                        "time_step": None,
                        "assignments": [],
                        "blocking": execution_plan.get("blocking"),
                    },
                    loop_dir / "executed_allocation_unit.json",
                )
        finally:
            if loop_qwen_chat is not None:
                from conceptgraph.vlm import close_vlm_chat

                close_vlm_chat(loop_qwen_chat)
        log_memory(f"loop {loop_index} planning end", args)

        if execution_plan is None:
            raise RuntimeError("Allocation did not return an Execution Plan")
        if execution_plan.get("state") == "blocked":
            blocking = deepcopy(execution_plan.get("blocking") or {})
            source_key = runtime_replan_source_key(
                task_graph_version=task_graph_version,
                trigger="allocation_blocked",
                evidence={
                    "graph_fingerprint": execution_plan.get("graph_fingerprint"),
                    "blocking": blocking,
                },
            )
            within_episode_budget = task_graph_replan_count < max(
                int(getattr(args, "max_task_graph_replans", 3)), 0
            )
            within_source_budget = (
                task_graph_replan_counts_by_source.get(source_key, 0)
                < max(int(getattr(args, "max_task_replans_per_source", 1)), 0)
            )
            if within_episode_budget and within_source_budget:
                replan_result = replan_datagen_task_graph(
                    root_task=args.task,
                    current_task_graph=original_task_graph,
                    scene_context=pre_task_subgraph["subgraph"],
                    agent_states=pre_stage["agent_states"],
                    completed=completed,
                    failed=failed,
                    trigger="allocation_blocked",
                    trigger_evidence={
                        "execution_plan": deepcopy(execution_plan),
                    },
                    task_graph_version=task_graph_version,
                    replan_index=task_graph_replan_count + 1,
                    output_dir=(
                        loop_dir
                        / "replanning"
                        / f"replan_{task_graph_replan_count + 1:03d}"
                    ),
                    args=args,
                )
                task_graph_replan_count += 1
                task_graph_replan_counts_by_source[source_key] = (
                    task_graph_replan_counts_by_source.get(source_key, 0) + 1
                )
            else:
                replan_result = {
                    "status": "skipped",
                    "trigger": "allocation_blocked",
                    "reason": (
                        "episode_replan_budget_exhausted"
                        if not within_episode_budget
                        else "source_replan_budget_exhausted"
                    ),
                }
            replan_event = {
                key: deepcopy(value)
                for key, value in replan_result.items()
                if key != "task_graph"
            }
            replan_event["source_key"] = source_key
            task_graph_replan_events.append(replan_event)
            save_json(replan_event, loop_dir / "task_graph_replan_result.json")

            if replan_result.get("status") == "success":
                original_task_graph = replan_result["task_graph"]
                task_graph_version = int(
                    original_task_graph.get("task_graph_version")
                    or task_graph_version + 1
                )
                task_ids = set(task_index(original_task_graph))
                completed = set()
                failed = set()
                retry_counts = {}
                loop_record = {
                    "loop_index": loop_index,
                    "state": "replanned",
                    "trigger": "allocation_blocked",
                    "active_task_graph": str(loop_dir / "active_task_graph.json"),
                    "allocation_request": str(loop_dir / "allocation_request.json"),
                    "execution_plan": str(loop_dir / "execution_plan.json"),
                    "executed_allocation_unit": str(loop_dir / "executed_allocation_unit.json"),
                    "task_graph_replanning": str(loop_dir / "task_graph_replan_result.json"),
                    "runtime_task_graph_after_replanning": replan_result[
                        "task_graph_path"
                    ],
                }
                save_json(loop_record, loop_dir / "loop_record.json")
                sample_record["loops"].append(loop_record)
                log(
                    f"sample {sample_index}: replaced Task Graph after "
                    "Allocation reported blocked"
                )
                continue

            termination = {
                "state": "blocked",
                "stage": "allocation_replanning",
                "loop_index": loop_index,
                "task_graph_version": task_graph_version,
                "graph_fingerprint": execution_plan.get("graph_fingerprint"),
                "blocking": blocking,
                "reason_code": (
                    "allocation_replan_failed"
                    if replan_result.get("status") == "failed"
                    else str(replan_result.get("reason") or "allocation_replan_unavailable")
                ),
                "replan_result": replan_event,
            }
            save_json(termination, loop_dir / "allocation_blocked.json")
            loop_record = {
                "loop_index": loop_index,
                "state": "blocked",
                "pre_stage": str(loop_dir / "pre_stage_info.json"),
                "pre_scenegraph": str(loop_dir / "pre_scenegraph_info.json"),
                "pre_agent_states": str(loop_dir / "pre_agent_states.json"),
                "pre_task_subgraph": str(loop_dir / "pre_task_subgraph.json"),
                "pre_task_subgraph_info": str(loop_dir / "pre_task_subgraph_info.json"),
                "active_task_graph": str(loop_dir / "active_task_graph.json"),
                "allocation_request": str(loop_dir / "allocation_request.json"),
                "execution_plan": str(loop_dir / "execution_plan.json"),
                "executed_allocation_unit": str(loop_dir / "executed_allocation_unit.json"),
                "allocation_blocked": str(loop_dir / "allocation_blocked.json"),
                "task_graph_replanning": str(loop_dir / "task_graph_replan_result.json"),
            }
            save_json(loop_record, loop_dir / "loop_record.json")
            sample_record["loops"].append(loop_record)
            sample_record["termination"] = termination
            log(
                f"sample {sample_index}: Allocation blocked "
                f"({blocking.get('code', blocking.get('reason_code', 'unknown'))}), "
                f"runtime replanning ended with {termination['reason_code']}"
            )
            break

        if execution is None:
            raise RuntimeError("dispatchable Execution Plan was not executed")
        video_recorder.capture(controller.last_event, {"stage": "after_execution", "loop_index": loop_index})

        post_stage = collect_stage(controller, sample_dir, f"loop_{loop_index:03d}_post", args)
        task_statuses = evaluate_task_statuses(
            assignments=assignments,
            execution=execution,
            pre_agent_states=pre_stage["agent_states"],
            post_agent_states=post_stage["agent_states"],
            retry_counts=retry_counts,
            max_task_retries=args.max_task_retries,
        )
        completed_this_loop: list[str] = []
        failed_this_loop: list[str] = []
        waiting_this_loop: list[str] = []
        for status in task_statuses:
            task_id = str(status["subtask_id"])
            state = str(status["status"])
            if state == SUCCESS:
                completed.add(task_id)
                completed_this_loop.append(task_id)
                retry_counts.pop(task_id, None)
                continue
            if state == FAILURE:
                failed.add(task_id)
                failed_this_loop.append(task_id)
                retry_counts.pop(task_id, None)
                continue
            retry_counts[task_id] = int(status.get("retry_count_after", retry_counts.get(task_id, 0)))
            waiting_this_loop.append(task_id)

        post_scenegraph, merged_scenegraph = update_stage_scenegraph_incrementally(
            pre_stage=pre_stage,
            post_stage=post_stage,
            current_scenegraph=current_scenegraph,
            execution=execution,
            output_dir=loop_dir / "merged_scenegraph",
            args=args,
        )
        post_task_subgraph = build_loop_task_subgraph(
            scenegraph_info=merged_scenegraph,
            tasks=unfinished_tasks,
            overall_task=args.task,
            output_path=loop_dir / "post_task_subgraph.json",
            args=args,
        )
        completion = {
            "task_statuses": task_statuses,
            "execution_metrics": {
                "execution_time_seconds": float(execution.get("execution_time_seconds") or 0.0),
                "macro_step_wall_time_seconds": float(execution.get("macro_step_wall_time_seconds") or 0.0),
                "simulation_time_seconds": execution.get("simulation_time_seconds"),
                "collided": bool(execution.get("collided")),
                "collision_count": int(execution.get("collision_count") or 0),
                "collided_objects": list(execution.get("collided_objects") or []),
            },
            "completed_this_loop": completed_this_loop,
            "failed_this_loop": failed_this_loop,
            "wait_retry_this_loop": waiting_this_loop,
            "completed_all": sorted(completed),
            "failed_all": sorted(failed),
            "remaining": sorted(task_ids - completed - failed),
            "terminal": sorted(completed | failed),
            "all_done": completed >= task_ids,
        }
        save_json(execution, loop_dir / "execution_trace.json")
        save_json({key: value for key, value in post_stage.items() if key != "agent_states"}, loop_dir / "post_stage_info.json")
        save_json(post_stage["agent_states"], loop_dir / "post_agent_states.json")
        save_json(post_scenegraph, loop_dir / "post_scenegraph_info.json")
        save_json(merged_scenegraph, loop_dir / "merged_scenegraph_info.json")
        save_json({key: value for key, value in post_task_subgraph.items() if key != "subgraph"}, loop_dir / "post_task_subgraph_info.json")
        save_json(task_statuses, loop_dir / "task_status.json")
        save_json(completion, loop_dir / "task_completion.json")

        loop_record = {
            "loop_index": loop_index,
            "pre_stage": str(loop_dir / "pre_stage_info.json"),
            "pre_scenegraph": str(loop_dir / "pre_scenegraph_info.json"),
            "pre_agent_states": str(loop_dir / "pre_agent_states.json"),
            "pre_task_subgraph": str(loop_dir / "pre_task_subgraph.json"),
            "pre_task_subgraph_info": str(loop_dir / "pre_task_subgraph_info.json"),
            "active_task_graph": str(loop_dir / "active_task_graph.json"),
            "allocation_request": str(loop_dir / "allocation_request.json"),
            "execution_plan": str(loop_dir / "execution_plan.json"),
            "executed_allocation_unit": str(loop_dir / "executed_allocation_unit.json"),
            "execution_trace": str(loop_dir / "execution_trace.json"),
            "post_stage": str(loop_dir / "post_stage_info.json"),
            "post_scenegraph": str(loop_dir / "post_scenegraph_info.json"),
            "post_agent_states": str(loop_dir / "post_agent_states.json"),
            "post_task_subgraph": str(loop_dir / "post_task_subgraph.json"),
            "post_task_subgraph_info": str(loop_dir / "post_task_subgraph_info.json"),
            "merged_scenegraph": str(Path(merged_scenegraph["scene_graph"])),
            "merged_scenegraph_info": str(loop_dir / "merged_scenegraph_info.json"),
            "task_status": str(loop_dir / "task_status.json"),
            "task_completion": str(loop_dir / "task_completion.json"),
        }

        if failed_this_loop:
            failure_evidence = {
                "failed_task_ids": sorted(failed_this_loop),
                "task_statuses": [
                    deepcopy(status)
                    for status in task_statuses
                    if str(status.get("status")) == FAILURE
                ],
            }
            source_key = runtime_replan_source_key(
                task_graph_version=task_graph_version,
                trigger="execution_failure",
                evidence=failure_evidence,
            )
            within_episode_budget = task_graph_replan_count < max(
                int(getattr(args, "max_task_graph_replans", 3)), 0
            )
            within_source_budget = (
                task_graph_replan_counts_by_source.get(source_key, 0)
                < max(int(getattr(args, "max_task_replans_per_source", 1)), 0)
            )
            if within_episode_budget and within_source_budget:
                replan_result = replan_datagen_task_graph(
                    root_task=args.task,
                    current_task_graph=original_task_graph,
                    scene_context=post_task_subgraph["subgraph"],
                    agent_states=post_stage["agent_states"],
                    completed=completed,
                    failed=failed,
                    trigger="execution_failure",
                    trigger_evidence={
                        **failure_evidence,
                        "execution": deepcopy(execution),
                    },
                    task_graph_version=task_graph_version,
                    replan_index=task_graph_replan_count + 1,
                    output_dir=(
                        loop_dir
                        / "replanning"
                        / f"replan_{task_graph_replan_count + 1:03d}"
                    ),
                    args=args,
                )
                task_graph_replan_count += 1
                task_graph_replan_counts_by_source[source_key] = (
                    task_graph_replan_counts_by_source.get(source_key, 0) + 1
                )
            else:
                replan_result = {
                    "status": "skipped",
                    "trigger": "execution_failure",
                    "reason": (
                        "episode_replan_budget_exhausted"
                        if not within_episode_budget
                        else "source_replan_budget_exhausted"
                    ),
                }
            replan_event = {
                key: deepcopy(value)
                for key, value in replan_result.items()
                if key != "task_graph"
            }
            replan_event["source_key"] = source_key
            task_graph_replan_events.append(replan_event)
            save_json(replan_event, loop_dir / "task_graph_replan_result.json")
            loop_record["task_graph_replanning"] = str(
                loop_dir / "task_graph_replan_result.json"
            )

            if replan_result.get("status") == "success":
                loop_record["state"] = "replanned"
                loop_record["trigger"] = "execution_failure"
                loop_record["runtime_task_graph_after_replanning"] = replan_result[
                    "task_graph_path"
                ]
                save_json(loop_record, loop_dir / "loop_record.json")
                sample_record["loops"].append(loop_record)
                current_scenegraph = merged_scenegraph
                original_task_graph = replan_result["task_graph"]
                task_graph_version = int(
                    original_task_graph.get("task_graph_version")
                    or task_graph_version + 1
                )
                task_ids = set(task_index(original_task_graph))
                completed = set()
                failed = set()
                retry_counts = {}
                log(
                    f"sample {sample_index}: replaced Task Graph after terminal "
                    "execution failure"
                )
                continue

            termination = {
                "state": "blocked",
                "stage": "execution_replanning",
                "loop_index": loop_index,
                "task_graph_version": task_graph_version,
                "failed_task_ids": sorted(failed_this_loop),
                "reason_code": (
                    "execution_replan_failed"
                    if replan_result.get("status") == "failed"
                    else str(replan_result.get("reason") or "execution_replan_unavailable")
                ),
                "replan_result": replan_event,
            }
            save_json(termination, loop_dir / "execution_replanning_blocked.json")
            loop_record["state"] = "blocked"
            loop_record["execution_replanning_blocked"] = str(
                loop_dir / "execution_replanning_blocked.json"
            )
            save_json(loop_record, loop_dir / "loop_record.json")
            sample_record["loops"].append(loop_record)
            sample_record["termination"] = termination
            current_scenegraph = merged_scenegraph
            break

        save_json(loop_record, loop_dir / "loop_record.json")
        sample_record["loops"].append(loop_record)
        current_scenegraph = merged_scenegraph

    final_stage = collect_stage(controller, sample_dir, "final", args)
    video_recorder.capture(controller.last_event, {"stage": "final"})
    video_path = video_recorder.close()
    sample_record["final"] = {
        "agent_states": str(Path(final_stage["stage_dir"]) / "agent_states.json"),
        "completed_tasks": sorted(completed),
        "failed_tasks": sorted(failed),
        "remaining_tasks": sorted(task_ids - completed - failed),
        "all_done": completed >= task_ids,
        "task_graph_version": task_graph_version,
        "task_graph_replan_count": task_graph_replan_count,
        "task_graph_replan_events": task_graph_replan_events,
    }
    if video_path is not None:
        sample_record["agent_operations_video"] = video_path
    save_json(sample_record, sample_dir / "sample_record.json")
    return sample_record


def main() -> None:
    args = build_parser().parse_args()
    start_xserver_if_needed(args)
    validate_cloud_rendering_environment(args)

    run_name = args.run_name or f"{args.scene_name}_{int(time.time())}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(vars(args) | {"output_dir": str(args.output_dir), "dataset_config": str(args.dataset_config)}, run_dir / "args.json")

    log(
        "starting AI2-THOR Controller "
        f"(scene={args.scene_name}, agents={args.agentnum}, platform={args.platform})"
    )
    controller = make_controller(args)
    records = []
    try:
        for sample_index in tqdm(range(args.episode), desc="trajectory samples"):
            records.append(run_sample(controller, sample_index, run_dir, args))
    finally:
        controller.stop()

    manifest = {
        "run_dir": str(run_dir),
        "num_samples": len(records),
        "samples": [str(run_dir / f"sample_{idx:06d}" / "sample_record.json") for idx in range(len(records))],
    }
    save_json(manifest, run_dir / "manifest.json")
    log(f"saved trajectory dataset: {run_dir}")


if __name__ == "__main__":
    main()
