"""
Small client for the AI2-THOR WebSocket bridge.

It can fetch one snapshot, stream observations for a few seconds, or send one
action to a selected agent.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import Any


try:
    import websockets
except ImportError as exc:  # pragma: no cover - depends on runtime env
    raise SystemExit("websockets is required: python -m pip install websockets") from exc


def strip_frame_payloads(payload: Any) -> Any:
    """Keep metadata readable when printing or writing compact JSON."""
    if isinstance(payload, dict):
        if "data" in payload and payload.get("encoding") in {"raw_base64", "png_base64"}:
            compact = dict(payload)
            compact["data"] = f"<base64:{len(payload['data'])} chars>"
            return compact
        return {key: strip_frame_payloads(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [strip_frame_payloads(item) for item in payload]
    return payload


def decode_raw_array(blob: dict[str, Any]) -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional local decode
        raise RuntimeError("Saving raw arrays requires numpy.") from exc

    raw = base64.b64decode(blob["data"])
    return np.frombuffer(raw, dtype=np.dtype(blob["dtype"])).reshape(blob["shape"])


def save_snapshot_arrays(snapshot: dict[str, Any], output_dir: Path) -> None:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional local decode
        raise RuntimeError("Saving arrays requires numpy.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    for agent in snapshot.get("agents") or []:
        agent_id = agent.get("agent_id", "0")
        for key in ("rgb", "depth", "instance"):
            blob = agent.get(key)
            if not isinstance(blob, dict):
                continue
            if blob.get("encoding") == "raw_base64":
                np.save(output_dir / f"agent_{agent_id}_{key}.npy", decode_raw_array(blob))
            elif blob.get("encoding") == "png_base64":
                (output_dir / f"agent_{agent_id}_{key}.png").write_bytes(base64.b64decode(blob["data"]))


async def send_request(uri: str, request: dict[str, Any]) -> dict[str, Any]:
    async with websockets.connect(uri, max_size=256 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps(request, ensure_ascii=False))
        return json.loads(await websocket.recv())


async def stream(uri: str, request: dict[str, Any], seconds: float) -> None:
    async with websockets.connect(uri, max_size=256 * 1024 * 1024) as websocket:
        await websocket.send(json.dumps(request, ensure_ascii=False))
        print(await websocket.recv())
        deadline = asyncio.get_event_loop().time() + seconds
        while asyncio.get_event_loop().time() < deadline:
            message = json.loads(await websocket.recv())
            agents = message.get("agents") or []
            poses = [{"agent_id": a.get("agent_id"), "pose": a.get("pose")} for a in agents]
            print(json.dumps({"sequence": message.get("sequence"), "poses": poses}, ensure_ascii=False))
        await websocket.send(json.dumps({"type": "stop_stream"}))
        print(await websocket.recv())


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Client for agents/ai2thor_ws_server.py")
    parser.add_argument("--uri", default="ws://127.0.0.1:8765")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument(
        "--compact-json",
        action="store_true",
        help="When writing --output-json, replace frame base64 payloads with size placeholders.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-rgb", action="store_true")
    parser.add_argument("--no-depth", action="store_true")
    parser.add_argument("--no-metadata", action="store_true")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("snapshot")

    step_parser = subparsers.add_parser("step")
    step_parser.add_argument("--agent-id", type=int, default=0)
    step_parser.add_argument("--action", required=True)
    step_parser.add_argument("--params", default="{}", help="JSON object with extra AI2-THOR action params")

    reset_parser = subparsers.add_parser("reset")
    reset_parser.add_argument("--scene", required=True)
    reset_parser.add_argument("--agent-count", type=int, default=2)

    stream_parser = subparsers.add_parser("stream")
    stream_parser.add_argument("--fps", type=float, default=2.0)
    stream_parser.add_argument("--seconds", type=float, default=5.0)
    return parser


def common_options(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "include_rgb": not args.no_rgb,
        "include_depth": not args.no_depth,
        "include_metadata": not args.no_metadata,
    }


def main() -> None:
    args = build_argparser().parse_args()
    command = args.command or "snapshot"
    request = common_options(args)

    if command == "step":
        request.update(
            {
                "type": "step",
                "agent_id": args.agent_id,
                "action": args.action,
                "params": json.loads(args.params),
            }
        )
    elif command == "reset":
        request.update({"type": "reset", "scene": args.scene, "agent_count": args.agent_count})
    elif command == "stream":
        request.update({"type": "start_stream", "fps": args.fps})
        asyncio.run(stream(args.uri, request, args.seconds))
        return
    else:
        request.update({"type": "snapshot"})

    response = asyncio.run(send_request(args.uri, request))
    if args.output_dir:
        save_snapshot_arrays(response, args.output_dir)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        json_payload = strip_frame_payloads(response) if args.compact_json else response
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)

    print(json.dumps(strip_frame_payloads(response), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
