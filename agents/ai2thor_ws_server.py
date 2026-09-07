"""
WebSocket bridge for remote AI2-THOR multi-agent observations.

Run this on the machine that has AI2-THOR/Unity available. Clients can request
RGB-D observations, poses, and metadata for all agents, or send actions to one
agent at a time.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
EMAS_ROOT = Path(__file__).resolve().parents[1]
AI2THOR_ROOT = EMAS_ROOT / "ai2thor"
for path in (WORKSPACE_ROOT, AI2THOR_ROOT, EMAS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit("numpy is required in the AI2-THOR runtime environment.") from exc

try:
    import websockets
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit(
        "websockets is required. Install it in the remote AI2-THOR environment with: "
        "python -m pip install websockets"
    ) from exc

try:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
except ImportError:
    from mwl.EMAS.ai2thor.ai2thor.controller import Controller
    from mwl.EMAS.ai2thor.ai2thor.platform import CloudRendering


def json_safe(value: Any) -> Any:
    """Convert AI2-THOR/numpy payloads into JSON-serializable values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return str(value)


def encode_array_raw(array: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(array)
    return {
        "encoding": "raw_base64",
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": base64.b64encode(array.tobytes()).decode("ascii"),
    }


def encode_array_png(array: np.ndarray, *, depth_scale: float) -> dict[str, Any]:
    try:
        import imageio.v2 as imageio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("image_encoding=png requires imageio: python -m pip install imageio") from exc

    payload = np.asarray(array)
    if payload.ndim == 2 and np.issubdtype(payload.dtype, np.floating):
        payload = np.round(payload * depth_scale).astype(np.uint16)
    elif payload.dtype != np.uint8 and payload.dtype != np.uint16:
        payload = np.clip(payload, 0, 255).astype(np.uint8)

    buf = io.BytesIO()
    imageio.imwrite(buf, payload, format="png")
    return {
        "encoding": "png_base64",
        "dtype": str(payload.dtype),
        "shape": list(payload.shape),
        "depth_scale": depth_scale if payload.ndim == 2 else None,
        "data": base64.b64encode(buf.getvalue()).decode("ascii"),
    }


def encode_array(array: Any, *, image_encoding: str, depth_scale: float) -> dict[str, Any] | None:
    if array is None:
        return None
    payload = np.asarray(array)
    if image_encoding == "png":
        return encode_array_png(payload, depth_scale=depth_scale)
    return encode_array_raw(payload)


def pose_from_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    agent = metadata.get("agent") or {}
    return {
        "position": json_safe(agent.get("position")),
        "rotation": json_safe(agent.get("rotation")),
        "cameraHorizon": json_safe(agent.get("cameraHorizon")),
        "isStanding": json_safe(agent.get("isStanding")),
    }


def event_list(event: Any) -> list[Any]:
    events = getattr(event, "events", None)
    if events is not None:
        return list(events)
    return [event]


class AI2ThorWebSocketBridge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.controller: Controller | None = None
        self.lock = asyncio.Lock()
        self.sequence = 0

    def start_controller(self) -> None:
        controller_kwargs: dict[str, Any] = {
            "scene": self.args.scene,
            "width": self.args.width,
            "height": self.args.height,
            "gridSize": self.args.grid_size,
            "visibilityDistance": self.args.visibility_distance,
            "fieldOfView": self.args.field_of_view,
            "renderDepthImage": True,
            "renderInstanceSegmentation": self.args.render_instance_segmentation,
            "agentCount": self.args.agent_count,
            "quality": self.args.quality,
        }
        if self.args.platform == "cloud":
            controller_kwargs["platform"] = CloudRendering
        if self.args.local_executable_path:
            controller_kwargs["local_executable_path"] = self.args.local_executable_path
        if self.args.x_display:
            controller_kwargs["x_display"] = self.args.x_display
        if self.args.gpu_device is not None:
            controller_kwargs["gpu_device"] = self.args.gpu_device

        self.controller = Controller(**controller_kwargs)

    def stop_controller(self) -> None:
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    def snapshot_from_last_event(
        self,
        *,
        include_rgb: bool,
        include_depth: bool,
        include_metadata: bool,
    ) -> dict[str, Any]:
        if self.controller is None or self.controller.last_event is None:
            raise RuntimeError("AI2-THOR controller is not running.")

        self.sequence += 1
        active_event = self.controller.last_event
        agents = []
        for fallback_id, agent_event in enumerate(event_list(active_event)):
            metadata = dict(getattr(agent_event, "metadata", {}) or {})
            agent = metadata.get("agent") or {}
            agent_id = agent.get("agentId", fallback_id)

            item: dict[str, Any] = {
                "agent_id": str(agent_id),
                "pose": pose_from_metadata(metadata),
                "lastAction": json_safe(metadata.get("lastAction")),
                "lastActionSuccess": json_safe(metadata.get("lastActionSuccess")),
            }
            if include_metadata:
                item["metadata"] = json_safe(metadata)
            if include_rgb:
                item["rgb"] = encode_array(
                    getattr(agent_event, "frame", None),
                    image_encoding=self.args.image_encoding,
                    depth_scale=self.args.depth_scale,
                )
            if include_depth:
                item["depth"] = encode_array(
                    getattr(agent_event, "depth_frame", None),
                    image_encoding=self.args.image_encoding,
                    depth_scale=self.args.depth_scale,
                )
            if self.args.render_instance_segmentation:
                item["instance"] = encode_array(
                    getattr(agent_event, "instance_segmentation_frame", None),
                    image_encoding=self.args.image_encoding,
                    depth_scale=self.args.depth_scale,
                )
            agents.append(item)

        return {
            "type": "snapshot",
            "ok": True,
            "sequence": self.sequence,
            "timestamp": time.time(),
            "scene": json_safe(getattr(self.controller, "scene", self.args.scene)),
            "agent_count": len(agents),
            "agents": agents,
        }

    async def handle_snapshot(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with self.lock:
            return self.snapshot_from_last_event(
                include_rgb=bool(payload.get("include_rgb", True)),
                include_depth=bool(payload.get("include_depth", True)),
                include_metadata=bool(payload.get("include_metadata", True)),
            )

    async def handle_step(self, payload: dict[str, Any]) -> dict[str, Any]:
        action = payload.get("action")
        if not action:
            raise ValueError("step request requires an 'action' field.")
        params = dict(payload.get("params") or {})
        if "agentId" not in params and "agent_id" in payload:
            params["agentId"] = int(payload["agent_id"])

        async with self.lock:
            if self.controller is None:
                raise RuntimeError("AI2-THOR controller is not running.")
            self.controller.step(action=action, **params)
            return self.snapshot_from_last_event(
                include_rgb=bool(payload.get("include_rgb", True)),
                include_depth=bool(payload.get("include_depth", True)),
                include_metadata=bool(payload.get("include_metadata", True)),
            )

    async def handle_reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        scene = payload.get("scene", self.args.scene)
        agent_count = int(payload.get("agent_count", self.args.agent_count))
        init_params = dict(payload.get("init_params") or {})
        init_params.setdefault("agentCount", agent_count)
        init_params.setdefault("renderDepthImage", True)
        init_params.setdefault("renderInstanceSegmentation", self.args.render_instance_segmentation)

        async with self.lock:
            if self.controller is None:
                raise RuntimeError("AI2-THOR controller is not running.")
            self.args.scene = scene
            self.args.agent_count = agent_count
            self.controller.reset(scene, **init_params)
            return self.snapshot_from_last_event(
                include_rgb=bool(payload.get("include_rgb", True)),
                include_depth=bool(payload.get("include_depth", True)),
                include_metadata=bool(payload.get("include_metadata", True)),
            )

    async def dispatch(self, payload: dict[str, Any]) -> dict[str, Any]:
        message_type = payload.get("type", "snapshot")
        if message_type in {"snapshot", "get_observations"}:
            return await self.handle_snapshot(payload)
        if message_type == "step":
            return await self.handle_step(payload)
        if message_type == "reset":
            return await self.handle_reset(payload)
        if message_type == "ping":
            return {"type": "pong", "ok": True, "timestamp": time.time()}
        raise ValueError(f"unsupported message type: {message_type}")

    async def client_loop(self, websocket: Any) -> None:
        stream_task: asyncio.Task | None = None

        async def stream(fps: float, options: dict[str, Any]) -> None:
            delay = 1.0 / max(fps, 0.1)
            while True:
                snapshot = await self.handle_snapshot(options)
                snapshot["type"] = "stream"
                await websocket.send(json.dumps(snapshot, ensure_ascii=False))
                await asyncio.sleep(delay)

        try:
            async for message in websocket:
                try:
                    payload = json.loads(message)
                    message_type = payload.get("type", "snapshot")
                    if message_type == "start_stream":
                        if stream_task is not None:
                            stream_task.cancel()
                        fps = float(payload.get("fps", self.args.stream_fps))
                        stream_task = asyncio.create_task(stream(fps, payload))
                        response = {"type": "start_stream", "ok": True, "fps": fps}
                    elif message_type == "stop_stream":
                        if stream_task is not None:
                            stream_task.cancel()
                            stream_task = None
                        response = {"type": "stop_stream", "ok": True}
                    else:
                        response = await self.dispatch(payload)
                except Exception as exc:
                    response = {
                        "type": "error",
                        "ok": False,
                        "error": str(exc),
                        "traceback": traceback.format_exc() if self.args.debug else None,
                    }
                await websocket.send(json.dumps(response, ensure_ascii=False))
        finally:
            if stream_task is not None:
                stream_task.cancel()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve AI2-THOR multi-agent RGB-D observations over WebSocket.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--agent-count", type=int, default=2)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--visibility-distance", type=float, default=1.5)
    parser.add_argument("--field-of-view", type=float, default=90.0)
    parser.add_argument("--quality", default="Very Low")
    parser.add_argument("--platform", choices=["default", "cloud"], default="cloud")
    parser.add_argument("--local-executable-path", default=None)
    parser.add_argument("--x-display", default=None)
    parser.add_argument("--gpu-device", type=int, default=None)
    parser.add_argument("--image-encoding", choices=["raw", "png"], default="raw")
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--render-instance-segmentation", action="store_true")
    parser.add_argument("--stream-fps", type=float, default=2.0)
    parser.add_argument("--max-size", type=int, default=256 * 1024 * 1024)
    parser.add_argument("--debug", action="store_true")
    return parser


async def async_main(args: argparse.Namespace) -> None:
    bridge = AI2ThorWebSocketBridge(args)
    bridge.start_controller()
    print(
        f"AI2-THOR WebSocket server listening on ws://{args.host}:{args.port} "
        f"(scene={args.scene}, agents={args.agent_count})",
        flush=True,
    )
    try:
        async with websockets.serve(
            bridge.client_loop,
            args.host,
            args.port,
            max_size=args.max_size,
            ping_interval=20,
            ping_timeout=20,
        ):
            await asyncio.Future()
    finally:
        bridge.stop_controller()


def main() -> None:
    args = build_argparser().parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

