#!/usr/bin/env python3
"""
AI2-THOR Multi-Agent Receiver Server
====================================
HTTP receiver using AI2-THOR's native multi-agent support through the
standard Controller:

    Controller(agentCount=N)
    controller.step(action=..., agentId=robot_id)

This keeps one shared Unity world and avoids RobotController/WSGI, which is
useful when WSGI hangs on Initialize(agentCount=N). Remote agents still talk to
this script over HTTP, so they do not need direct Unity access.
"""

from __future__ import annotations

import argparse
import base64
import heapq
import json
import math
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

try:
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    from ai2thor.server import MultiAgentEvent
except ModuleNotFoundError:
    Controller = None
    CloudRendering = None

    class MultiAgentEvent:  # type: ignore[no-redef]
        pass


DEFAULT_AGENT_Y = 0.900999128818512

DEFAULT_GRID_SIZE = 0.25
DEFAULT_GRID_EPSILON = 0.03
DEFAULT_ROTATE_STEP_DEGREES = 90.0
DEFAULT_YAW_TOLERANCE_DEGREES = 1e-3
DEFAULT_OBJECT_MIN_DISTANCE = 0.75
DEFAULT_HELD_OBJECT_MIN_DISTANCE = 1.0


class NavigationPlanningError(ValueError):
    def __init__(self, error_code: str, message: str):
        super().__init__(message)
        self.error_code = error_code


def _position_dict(value: Any, *, field: str = "position") -> dict[str, float]:
    if not isinstance(value, dict):
        raise NavigationPlanningError("invalid_target", f"{field} must be an object with x/y/z")
    if "x" not in value or "z" not in value:
        raise NavigationPlanningError("invalid_target", f"{field} must contain x and z")
    try:
        return {
            "x": float(value["x"]),
            "y": float(value.get("y", DEFAULT_AGENT_Y)),
            "z": float(value["z"]),
        }
    except (TypeError, ValueError) as exc:
        raise NavigationPlanningError("invalid_target", f"{field} coordinates must be numeric") from exc


def _node_for_position(position: dict[str, Any], precision: int = 3) -> tuple[float, float]:
    return (round(float(position["x"]), precision), round(float(position["z"]), precision))


def _distance_xz(a: dict[str, Any], b: dict[str, Any]) -> float:
    dx = float(a["x"]) - float(b["x"])
    dz = float(a["z"]) - float(b["z"])
    return math.sqrt(dx * dx + dz * dz)


def find_nearest_position(positions: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, float] | None:
    if not positions:
        return None
    return dict(min(positions, key=lambda pos: _distance_xz(pos, target)))


def choose_reachable_goal_candidates(
    reachable: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    min_distance: float = 0.0,
    max_distance: float | None = None,
) -> list[dict[str, float]]:
    candidates: list[dict[str, Any]] = []
    for position in reachable:
        try:
            distance = _distance_xz(position, target)
        except (KeyError, TypeError, ValueError):
            continue
        if distance < min_distance:
            continue
        if max_distance is not None and distance > max_distance:
            continue
        candidates.append(position)
    return [dict(pos) for pos in sorted(candidates, key=lambda pos: _distance_xz(pos, target))]


def choose_reachable_goal(
    reachable: list[dict[str, Any]],
    target: dict[str, Any],
    *,
    min_distance: float = 0.0,
    max_distance: float | None = None,
) -> dict[str, float] | None:
    candidates = choose_reachable_goal_candidates(
        reachable,
        target,
        min_distance=min_distance,
        max_distance=max_distance,
    )
    return candidates[0] if candidates else None


def build_reachable_graph(
    positions: list[dict[str, Any]],
    *,
    grid_size: float = DEFAULT_GRID_SIZE,
    epsilon: float = DEFAULT_GRID_EPSILON,
) -> tuple[dict[tuple[float, float], list[tuple[tuple[float, float], float]]], dict[tuple[float, float], dict[str, float]]]:
    node_positions: dict[tuple[float, float], dict[str, float]] = {}
    for position in positions:
        try:
            clean = _position_dict(position)
        except NavigationPlanningError:
            continue
        node_positions[_node_for_position(clean)] = clean

    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]] = {
        node: [] for node in node_positions
    }
    nodes = list(node_positions)
    for index, node in enumerate(nodes):
        pos = node_positions[node]
        for other in nodes[index + 1:]:
            other_pos = node_positions[other]
            dx = abs(pos["x"] - other_pos["x"])
            dz = abs(pos["z"] - other_pos["z"])
            horizontal_neighbor = dx <= grid_size + epsilon and dz <= epsilon and dx > epsilon
            vertical_neighbor = dz <= grid_size + epsilon and dx <= epsilon and dz > epsilon
            if not (horizontal_neighbor or vertical_neighbor):
                continue
            cost = _distance_xz(pos, other_pos)
            graph[node].append((other, cost))
            graph[other].append((node, cost))
    return graph, node_positions


def nodes_within_radius(
    node_positions: dict[tuple[float, float], dict[str, float]],
    center: dict[str, Any],
    *,
    radius: float,
) -> set[tuple[float, float]]:
    if radius <= 0:
        return set()
    return {
        node
        for node, position in node_positions.items()
        if _distance_xz(position, center) <= radius
    }


def remove_graph_nodes(
    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
    blocked_nodes: set[tuple[float, float]],
    *,
    preserve_nodes: set[tuple[float, float]] | None = None,
) -> dict[tuple[float, float], list[tuple[tuple[float, float], float]]]:
    preserve_nodes = preserve_nodes or set()
    blocked = blocked_nodes - preserve_nodes
    return {
        node: [(neighbor, cost) for neighbor, cost in edges if neighbor not in blocked]
        for node, edges in graph.items()
        if node not in blocked
    }


def _canonical_edge(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (start, end) if start <= end else (end, start)


def remove_graph_edges(
    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
    blocked_edges: set[tuple[tuple[float, float], tuple[float, float]]],
) -> dict[tuple[float, float], list[tuple[tuple[float, float], float]]]:
    blocked = {_canonical_edge(start, end) for start, end in blocked_edges}
    return {
        node: [
            (neighbor, cost)
            for neighbor, cost in edges
            if _canonical_edge(node, neighbor) not in blocked
        ]
        for node, edges in graph.items()
    }


def serialize_graph_node(node: tuple[float, float]) -> dict[str, float]:
    return {"x": node[0], "z": node[1]}


def serialize_graph_edge(edge: tuple[tuple[float, float], tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {"from": serialize_graph_node(edge[0]), "to": serialize_graph_node(edge[1])}


def failed_movement_edge(
    failed_action: dict[str, Any] | None,
    *,
    grid_size: float = DEFAULT_GRID_SIZE,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    if not isinstance(failed_action, dict):
        return None
    action = str(failed_action.get("action") or "")
    yaw_offset_by_action = {
        "MoveAhead": 0.0,
        "MoveBack": 180.0,
        "MoveRight": 90.0,
        "MoveLeft": -90.0,
    }
    if action not in yaw_offset_by_action:
        return None
    pose = failed_action.get("robot_pose") if isinstance(failed_action.get("robot_pose"), dict) else None
    agent = failed_action.get("agent") if isinstance(failed_action.get("agent"), dict) else None
    position = (pose or {}).get("position") or (agent or {}).get("position")
    rotation = (pose or {}).get("rotation") or (agent or {}).get("rotation")
    if not isinstance(position, dict):
        return None
    try:
        start_position = _position_dict(position, field="failed_action.position")
        current_yaw = float((rotation or {}).get("y", 0.0))
        move_yaw = (current_yaw + yaw_offset_by_action[action]) % 360.0
        radians = math.radians(move_yaw)
        end_position = {
            "x": start_position["x"] + math.sin(radians) * float(grid_size),
            "y": start_position["y"],
            "z": start_position["z"] + math.cos(radians) * float(grid_size),
        }
    except (TypeError, ValueError, NavigationPlanningError):
        return None
    return _canonical_edge(_node_for_position(start_position), _node_for_position(end_position))


def parse_navigation_blocker(failed_action: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(failed_action, dict):
        return {"failure_class": "unknown_blocker"}
    error = str(failed_action.get("error") or "")
    agent_match = re.search(r"Agent\s+(\d+)\s+is\s+blocking\s+Agent\s+(\d+)", error, re.I)
    if agent_match:
        return {
            "failure_class": "agent_blocker",
            "blocking_robot_id": int(agent_match.group(1)),
            "blocked_robot_id": int(agent_match.group(2)),
            "error": error,
        }
    object_match = re.search(r"([^.;]+?)\s+is\s+blocking\s+(?:the\s+)?Agent(?:\s+(\d+))?", error, re.I)
    if object_match:
        blocker = object_match.group(1).strip()
        if not blocker.lower().startswith("agent "):
            blocked_robot_id = object_match.group(2)
            fallback_robot_id = failed_action.get("robot_id") if isinstance(failed_action, dict) else None
            try:
                blocked_robot_id = int(blocked_robot_id if blocked_robot_id is not None else fallback_robot_id)
            except (TypeError, ValueError):
                blocked_robot_id = None
            return {
                "failure_class": "object_blocker",
                "blocking_object_id": blocker,
                "blocked_robot_id": blocked_robot_id,
                "error": error,
            }
    lowered = error.lower()
    if "wall" in lowered:
        return {"failure_class": "wall_blocker", "error": error}
    if "blocking" in lowered or "collid" in lowered or "obstruct" in lowered:
        return {"failure_class": "unknown_blocker", "error": error}
    return {"failure_class": "execution_failure", "error": error}


def astar_path(
    graph: dict[tuple[float, float], list[tuple[tuple[float, float], float]]],
    start: tuple[float, float],
    goal: tuple[float, float],
) -> list[tuple[float, float]] | None:
    if start not in graph or goal not in graph:
        return None
    open_heap: list[tuple[float, tuple[float, float]]] = [(0.0, start)]
    came_from: dict[tuple[float, float], tuple[float, float]] = {}
    g_score: dict[tuple[float, float], float] = {start: 0.0}
    closed: set[tuple[float, float]] = set()

    def heuristic(node: tuple[float, float]) -> float:
        return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        closed.add(current)
        for neighbor, edge_cost in graph.get(current, []):
            tentative = g_score[current] + edge_cost
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            heapq.heappush(open_heap, (tentative + heuristic(neighbor), neighbor))
    return None


def signed_angle_delta(target_yaw: float, current_yaw: float) -> float:
    return (target_yaw - current_yaw + 180.0) % 360.0 - 180.0


def _yaw_for_step(current: dict[str, Any], next_position: dict[str, Any]) -> float:
    dx = float(next_position["x"]) - float(current["x"])
    dz = float(next_position["z"]) - float(current["z"])
    if abs(dx) < 1e-6 and abs(dz) < 1e-6:
        return 0.0
    return math.degrees(math.atan2(dx, dz)) % 360.0


def yaw_rotation_actions(
    *,
    current_yaw: float,
    target_yaw: float,
    rotate_step_degrees: float = DEFAULT_ROTATE_STEP_DEGREES,
    yaw_tolerance_degrees: float = DEFAULT_YAW_TOLERANCE_DEGREES,
) -> list[dict[str, Any]]:
    if rotate_step_degrees <= 0:
        raise NavigationPlanningError("invalid_target", "rotate_step_degrees must be positive")
    if yaw_tolerance_degrees < 0:
        raise NavigationPlanningError("invalid_target", "yaw_tolerance_degrees must be non-negative")
    delta = signed_angle_delta(target_yaw, current_yaw)
    abs_delta = abs(delta)
    if abs_delta <= yaw_tolerance_degrees:
        return []
    turn_action = "RotateRight" if delta > 0 else "RotateLeft"
    turns = int(round(abs_delta / rotate_step_degrees))
    snapped_degrees = turns * rotate_step_degrees
    if turns > 0 and abs(abs_delta - snapped_degrees) <= yaw_tolerance_degrees:
        return [{"action": turn_action} for _ in range(turns)]
    return [{"action": turn_action, "degrees": abs_delta}]


def horizon_rotation_actions(
    *,
    current_horizon: float,
    target_horizon: float,
    horizon_tolerance_degrees: float = DEFAULT_YAW_TOLERANCE_DEGREES,
) -> list[dict[str, Any]]:
    """Return a single precise camera action for an AI2-THOR horizon."""

    delta = float(target_horizon) - float(current_horizon)
    if abs(delta) <= horizon_tolerance_degrees:
        return []
    degrees = round(abs(delta), 1)
    return [{"action": "LookDown" if delta > 0 else "LookUp", "degrees": degrees}]


def path_to_actions(
    path: list[dict[str, Any]],
    *,
    current_yaw: float,
    rotate_step_degrees: float = DEFAULT_ROTATE_STEP_DEGREES,
    yaw_tolerance_degrees: float = DEFAULT_YAW_TOLERANCE_DEGREES,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    yaw = float(current_yaw) % 360.0
    for current, next_position in zip(path, path[1:]):
        target_yaw = _yaw_for_step(current, next_position)
        actions.extend(
            yaw_rotation_actions(
                current_yaw=yaw,
                target_yaw=target_yaw,
                rotate_step_degrees=rotate_step_degrees,
                yaw_tolerance_degrees=yaw_tolerance_degrees,
            )
        )
        yaw = target_yaw
        actions.append({"action": "MoveAhead"})
    return actions


def held_object_safe_navigation_actions(
    actions: list[dict[str, Any]],
    *,
    holding_object: bool,
    default_rotate_degrees: float = DEFAULT_ROTATE_STEP_DEGREES,
    max_rotate_degrees: float = 15.0,
) -> list[dict[str, Any]]:
    """Use small forced turns while carrying an object to avoid hand-object collisions."""

    if not holding_object:
        return [dict(action) for action in actions]
    if max_rotate_degrees <= 0:
        raise NavigationPlanningError(
            "invalid_target",
            "held_max_rotate_degrees must be positive",
        )
    safe_actions: list[dict[str, Any]] = []
    for raw_action in actions:
        action = dict(raw_action)
        if action.get("action") not in {"RotateLeft", "RotateRight"}:
            safe_actions.append(action)
            continue
        try:
            degrees = abs(float(action.get("degrees", default_rotate_degrees)))
        except (TypeError, ValueError) as exc:
            raise NavigationPlanningError(
                "invalid_target",
                "rotation degrees must be numeric",
            ) from exc
        if degrees <= 0:
            continue
        turn_count = max(1, int(math.ceil(degrees / max_rotate_degrees)))
        turn_degrees = degrees / turn_count
        for _ in range(turn_count):
            safe_turn = dict(action)
            safe_turn["degrees"] = turn_degrees
            safe_turn["forceAction"] = True
            safe_actions.append(safe_turn)
    return safe_actions


def find_object_target(payload: dict[str, Any], state: dict[str, Any], robot_position: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    if "target_position" in payload:
        target = _position_dict(payload.get("target_position"), field="target_position")
        return target, {"kind": "position", "position": target}

    objects = state.get("objects", []) if isinstance(state, dict) else []
    if not isinstance(objects, list):
        objects = []
    object_id = payload.get("object_id") or payload.get("objectId")
    object_type = payload.get("object_type") or payload.get("objectType")
    matches: list[dict[str, Any]] = []
    if isinstance(object_id, str) and object_id.strip():
        matches = [obj for obj in objects if obj.get("id") == object_id or obj.get("objectId") == object_id]
    elif isinstance(object_type, str) and object_type.strip():
        normalized = object_type.lower()
        matches = [obj for obj in objects if str(obj.get("type") or obj.get("objectType") or "").lower() == normalized]
    else:
        raise NavigationPlanningError("invalid_target", "provide target_position, object_id, or object_type")

    positioned = [obj for obj in matches if isinstance(obj.get("position"), dict)]
    if not positioned:
        raise NavigationPlanningError("target_not_found", "target object was not found with a usable position")
    selected = min(positioned, key=lambda obj: _distance_xz(obj["position"], robot_position))
    target = _position_dict(selected["position"], field="object.position")
    return target, {
        "kind": "object",
        "object_id": selected.get("id") or selected.get("objectId"),
        "object_type": selected.get("type") or selected.get("objectType"),
        "position": target,
        "visible": selected.get("visible"),
        "distance": selected.get("distance"),
    }

_CV2 = None
_CV2_IMPORT_ERROR = None


def get_cv2():
    global _CV2, _CV2_IMPORT_ERROR
    if _CV2 is not None:
        return _CV2
    if _CV2_IMPORT_ERROR is not None:
        return None
    try:
        import cv2
    except Exception as exc:
        _CV2_IMPORT_ERROR = exc
        return None
    _CV2 = cv2
    return _CV2


def log_event(channel: str, message: str = "", *, file=None, blank_before: bool = False):
    stream = file or sys.stdout
    if blank_before:
        print("", file=stream, flush=True)
    for line in (str(message).splitlines() or [""]):
        print(f"[{channel}] {line}", file=stream, flush=True)


@dataclass
class RobotState:
    robot_id: int
    name: str
    position: dict
    rotation: dict
    horizon: float = 0.0
    task: str = ""
    last_action: str = "Pass"
    last_success: bool = True
    last_error: Optional[str] = None
    last_event: object = None
    inventory: list[dict] = field(default_factory=list)
    action_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        held_object = self.inventory[0] if self.inventory else None
        return {
            "robot_id": self.robot_id,
            "name": self.name,
            "position": self.position,
            "rotation": self.rotation,
            "horizon": self.horizon,
            "inventory": self.inventory,
            "held_object": held_object,
            "task": self.task,
            "last_action": self.last_action,
            "last_success": self.last_success,
            "last_error": self.last_error,
            "action_count": self.action_count,
            "controller": "native_standard_controller",
        }


class NativeControllerThorServer:
    def __init__(
        self,
        scene: str = "FloorPlan1",
        robot_count: int = 2,
        headless: bool = True,
        width: int = 600,
        height: int = 600,
        robot0_dx: float = 0.0,
        robot0_dz: float = 0.0,
        robot0_left: float = 0.0,
        robot0_right: float = 0.0,
        robot0_back: float = 0.0,
        robot0_dyaw: float = 0.0,
        robot0_at_fridge: bool = False,
        robot0_fridge_distance: float = 1.0,
        render_action_frames: bool = False,
    ):
        self.scene = scene
        self.robot_count = max(1, robot_count)
        self.headless = headless
        self.width = width
        self.height = height
        self.robot0_dx = robot0_dx
        self.robot0_dz = robot0_dz
        self.robot0_left = robot0_left
        self.robot0_right = robot0_right
        self.robot0_back = robot0_back
        self.robot0_dyaw = robot0_dyaw
        self.robot0_at_fridge = robot0_at_fridge
        self.robot0_fridge_distance = robot0_fridge_distance
        self.render_action_frames = bool(render_action_frames)

        self.controller = None
        self.robots: list[RobotState] = []
        self.lock = threading.RLock()
        self._step_count = 0
        self._window_names: dict[int, str] = {}
        self.action_callback = None

    def start(self):
        kwargs = dict(
            scene=self.scene,
            width=self.width,
            height=self.height,
            agentCount=self.robot_count,
            port=0,
        )
        if self.headless:
            kwargs["platform"] = CloudRendering

        log_event(
            "SIM IN",
            f"Start standard Controller(scene={self.scene}, robots={self.robot_count}, headless={self.headless})",
        )
        if Controller is None:
            raise RuntimeError("ai2thor is not installed; install ai2thor to start the receiver server")
        self.controller = Controller(**kwargs)
        self._validate_agent_count(self.controller.last_event)
        self._setup_robots()
        log_event("SIM OUT", "Native standard Controller multi-agent ready")

    def stop(self):
        with self.lock:
            if self.controller is not None:
                log_event("SIM IN", "Stop standard Controller")
                self.controller.stop()
                self.controller = None

    def reset(self, scene: Optional[str] = None, robot_count: Optional[int] = None):
        with self.lock:
            if scene:
                self.scene = scene
            if robot_count is not None:
                self.robot_count = max(1, int(robot_count))
            self.stop()
            self._step_count = 0
            self.start()

    def _validate_agent_count(self, event):
        events = self._events_from(event)
        meta_agents = event.metadata.get("agents", []) if hasattr(event, "metadata") else []
        count = max(len(events), len(meta_agents), 1)
        log_event("SIM OUT", f"Controller agents={count} requested={self.robot_count}")
        if count < self.robot_count:
            raise RuntimeError(
                f"Controller created {count} agents, requested {self.robot_count}. "
                "This AI2-THOR build may not support agentCount with standard Controller."
            )

    def _setup_robots(self):
        positions = self._select_spawn_positions(self.robot_count)
        if self.robot0_at_fridge and positions:
            positions[0] = self._select_position_in_front_of_object("Fridge", positions[0])
            positions = self._avoid_spawn_overlap(positions)

        center = self._position_center(positions)
        self.robots = []
        log_event("SIM OUT", "Initial native Controller agent positions:")

        for robot_id in range(self.robot_count):
            pos = dict(positions[robot_id])
            yaw = self._yaw_to_target(pos, center)
            if robot_id == 0 and self.robot0_at_fridge:
                fridge_pos = self._find_object_position("Fridge")
                if fridge_pos:
                    yaw = self._yaw_to_target(pos, fridge_pos)
            if robot_id == 0:
                pos, yaw = self._apply_robot0_initial_offset(pos, yaw)

            robot = RobotState(
                robot_id=robot_id,
                name=f"Robot{robot_id}",
                position=pos,
                rotation={"x": 0, "y": yaw, "z": 0},
                task=f"Native standard Controller agent slot {robot_id}",
            )
            self.robots.append(robot)
            event = self._controller_step(
                {
                    "action": "TeleportFull",
                    "agentId": robot_id,
                    "position": robot.position,
                    "rotation": robot.rotation,
                    "horizon": robot.horizon,
                    "standing": True,
                }
            )
            agent_event = self._event_for_robot(event, robot_id)
            robot.last_event = agent_event
            self._update_robot_pose_from_metadata(robot, agent_event.metadata)
            log_event(
                "SIM OUT",
                f"  {robot.name}: pos=({robot.position['x']:.2f}, "
                f"{robot.position.get('y', DEFAULT_AGENT_Y):.2f}, {robot.position['z']:.2f}), "
                f"yaw={robot.rotation.get('y', 0):.1f}",
            )

    def resolve_robot(self, robot_ref=None) -> RobotState:
        if robot_ref is None or robot_ref == "":
            return self.robots[0]
        if isinstance(robot_ref, str):
            robot_ref = robot_ref.strip()
            for robot in self.robots:
                if robot.name.lower() == robot_ref.lower():
                    return robot
            if robot_ref.lower().startswith("robot") and robot_ref[5:].isdigit():
                robot_ref = int(robot_ref[5:])
        try:
            robot_id = int(robot_ref)
        except (TypeError, ValueError):
            raise ValueError(f"unknown robot_id: {robot_ref!r}")
        if robot_id < 0 or robot_id >= len(self.robots):
            raise ValueError(f"robot_id out of range: {robot_id} (0..{len(self.robots)-1})")
        return self.robots[robot_id]

    def observe(self, robot_ref=None, render_image: bool = True) -> dict:
        with self.lock:
            robot = self.resolve_robot(robot_ref)
            event = self._controller_step({"action": "Pass", "agentId": robot.robot_id, "renderImage": render_image})
            self._refresh_all_robot_states(event)
            agent_event = self._event_for_robot(event, robot.robot_id)
            data = {
                "status": "success",
                "robot_id": robot.robot_id,
                "robot": robot.to_dict(),
                "objects": self._objects_from_metadata(agent_event.metadata),
                "metadata": agent_event.metadata,
            }
            if render_image:
                image_b64 = self._encode_image(agent_event)
                if image_b64:
                    data["image_base64"] = image_b64
            return data

    def execute(self, robot_ref, action: str, render_image: bool = False, **kwargs) -> dict:
        with self.lock:
            robot = self.resolve_robot(robot_ref)
            before_pose = self._robot_pose_snapshot(robot)
            before_meta = getattr(robot.last_event, "metadata", {}) if robot.last_event is not None else {}
            result = {
                "robot_id": robot.robot_id,
                "robot_name": robot.name,
                "action": action,
                "success": False,
                "error": None,
                "agent": None,
                "inventory": [],
                "controller": "native_standard_controller",
                "robot_pose": before_pose,
                "robot_pose_changed": False,
                "interacted_objects": [],
            }
            try:
                action_dict = dict(kwargs)
                action_dict["action"] = action
                action_dict["agentId"] = robot.robot_id
                action_dict["renderImage"] = render_image
                event = self._controller_step(action_dict)
                self._refresh_all_robot_states(event)
                agent_event = self._event_for_robot(event, robot.robot_id)
                meta = agent_event.metadata
                success = meta.get("lastActionSuccess", False)
                result["success"] = success
                if not success:
                    result["error"] = meta.get("errorMessage", "unknown error")
                result["agent"] = meta.get("agent", {})
                result["inventory"] = self._inventory_from_metadata(meta)
                robot.last_event = agent_event
                robot.last_action = action
                robot.last_success = success
                robot.last_error = result["error"]
                robot.action_count += 1
                self._step_count += 1
                self._update_robot_pose_from_metadata(robot, meta)
                result["robot"] = robot.to_dict()
                result["robot_pose"] = self._robot_pose_snapshot(robot)
                result["robot_pose_changed"] = self._pose_changed(before_pose, result["robot_pose"])
                if result["robot_pose_changed"]:
                    result["robot_pose_delta"] = {
                        "before": before_pose,
                        "after": result["robot_pose"],
                    }
                result["interacted_objects"] = self._interacted_objects_after_action(
                    action,
                    kwargs,
                    before_meta,
                    meta,
                )
                result["held_object"] = robot.inventory[0] if robot.inventory else None
                collided, collided_objects, collision_source = self._collision_from_metadata(meta, action)
                result["collided"] = collided
                result["collided_objects"] = collided_objects
                result["collision_source"] = collision_source
                callback = getattr(self, "action_callback", None)
                if callback is not None:
                    try:
                        callback(
                            event,
                            {
                                "action": action,
                                "robot_id": robot.robot_id,
                                "agent_id": robot.robot_id,
                                "params": dict(kwargs),
                                "lastActionSuccess": bool(success),
                                "errorMessage": result.get("error"),
                                "wall_time_seconds": None,
                                "simulation_time_seconds": None,
                                "collided": collided,
                                "collided_objects": collided_objects,
                                "collision_source": collision_source,
                            },
                        )
                    except Exception as exc:
                        log_event("SIM OUT", f"WARNING action_callback failed: {type(exc).__name__}: {exc}")
                if render_image:
                    image_b64 = self._encode_image(agent_event)
                    if image_b64:
                        result["image_base64"] = image_b64
            except ValueError as exc:
                result["error"] = str(exc)[:200]
                robot.last_success = False
                robot.last_error = result["error"]
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
                robot.last_success = False
                robot.last_error = result["error"]
            return result

    def execute_batch(
        self,
        actions: list[dict],
        default_robot_ref=None,
        render_image: bool = False,
        stop_on_failure: bool = True,
    ) -> dict:
        results = []
        all_succeeded = True
        last_robot_ref = default_robot_ref
        for i, act in enumerate(actions):
            if not isinstance(act, dict):
                results.append({"index": i, "action": str(act), "success": False, "error": "action must be a dict"})
                all_succeeded = False
                if stop_on_failure:
                    break
                continue
            act = dict(act)
            robot_ref = default_robot_ref
            for key in ("robot_id", "robot", "agent_id"):
                if key in act:
                    robot_ref = act.pop(key)
                    break
            action_name = act.pop("action", None)
            if action_name is None:
                results.append({"index": i, "robot_id": robot_ref, "success": False, "error": "missing action"})
                all_succeeded = False
                if stop_on_failure:
                    break
                continue
            try:
                item = self.execute(robot_ref, action_name, render_image=render_image, **act)
            except ValueError as exc:
                item = {"robot_id": robot_ref, "action": action_name, "success": False, "error": str(exc)}
            item["index"] = i
            results.append(item)
            if "robot_id" in item:
                last_robot_ref = item["robot_id"]
            if not item.get("success"):
                all_succeeded = False
                if stop_on_failure:
                    break
        if all_succeeded and len(results) == len(actions):
            status = "success"
        elif any(item.get("success") for item in results):
            status = "partial"
        else:
            status = "failed"
        return {"status": status, "results": results, "state": self._safe_capture_state(last_robot_ref)}

    def capture_state(self, robot_ref=None, render_image: bool = False) -> dict:
        with self.lock:
            selected = self.resolve_robot(robot_ref) if robot_ref is not None else self.robots[0]
            event = self._controller_step({"action": "Pass", "agentId": selected.robot_id, "renderImage": render_image})
            self._refresh_all_robot_states(event)
            agent_event = self._event_for_robot(event, selected.robot_id)
            meta = agent_event.metadata
            state = {
                "sceneName": meta.get("sceneName", self.scene),
                "step": self._step_count,
                "selected_robot_id": selected.robot_id,
                "controller_mode": "native_standard_controller_shared_unity",
                "agent": {
                    "position": selected.position,
                    "rotation": selected.rotation,
                    "horizon": selected.horizon,
                },
                "robots": [robot.to_dict() for robot in self.robots],
                "inventory": self._inventory_from_metadata(meta),
                "objects": self._objects_from_metadata(meta),
            }
            state["num_objects"] = len(state["objects"])
            if render_image:
                image_b64 = self._encode_image(agent_event)
                if image_b64:
                    state["image_base64"] = image_b64
            return state

    def _safe_capture_state(self, robot_ref=None, render_image: bool = False) -> dict:
        try:
            return self.capture_state(robot_ref, render_image=render_image)
        except Exception as exc:
            log_event(
                "SIM OUT",
                f"WARNING capture_state failed: {type(exc).__name__}: {exc}",
            )
            return {
                "status": "unavailable",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

    @staticmethod
    def _collision_from_metadata(meta: dict, action: str) -> tuple[bool, list[str], str | None]:
        collided_objects = [str(item) for item in meta.get("collidedObjects") or [] if item]
        if bool(meta.get("collided")) or collided_objects:
            return True, collided_objects, "ai2thor_metadata"
        error_message = str(meta.get("errorMessage") or "")
        lowered = error_message.lower()
        navigation_action = action in {"MoveAhead", "MoveBack", "MoveLeft", "MoveRight", "Teleport", "TeleportFull"}
        if navigation_action and not bool(meta.get("lastActionSuccess", True)) and ("blocking" in lowered or "collid" in lowered):
            return True, collided_objects, "error_message"
        return False, collided_objects, None

    def _controller_step(self, action: dict):
        with self.lock:
            controller_action = dict(action)
            if bool(getattr(self, "render_action_frames", False)):
                controller_action["renderImage"] = True
            return self.controller.step(controller_action)

    @staticmethod
    def _events_from(event):
        return event.events if isinstance(event, MultiAgentEvent) else [event]

    def _event_for_robot(self, event, robot_id: int):
        events = self._events_from(event)
        if robot_id >= len(events):
            return events[0]
        return events[robot_id]

    def _refresh_all_robot_states(self, event):
        events = self._events_from(event)
        for robot_id, agent_event in enumerate(events[: len(self.robots)]):
            robot = self.robots[robot_id]
            robot.last_event = agent_event
            self._update_robot_pose_from_metadata(robot, agent_event.metadata)

    def _update_robot_pose_from_metadata(self, robot: RobotState, meta: dict):
        agent = meta.get("agent", {})
        if agent.get("position"):
            robot.position = agent["position"]
        if agent.get("rotation"):
            robot.rotation = agent["rotation"]
        if "cameraHorizon" in agent:
            robot.horizon = agent["cameraHorizon"]
        robot.inventory = self._inventory_from_metadata(meta)

    @staticmethod
    def _inventory_from_metadata(meta: dict) -> list[dict]:
        return [
            {"objectId": item["objectId"], "objectType": item.get("objectType", "")}
            for item in meta.get("inventoryObjects", [])
        ]

    @staticmethod
    def _robot_pose_snapshot(robot: RobotState) -> dict:
        return {
            "position": dict(robot.position or {}),
            "rotation": dict(robot.rotation or {}),
            "horizon": robot.horizon,
        }

    @classmethod
    def _pose_changed(cls, before: dict, after: dict, eps: float = 1e-5) -> bool:
        return (
            cls._dict_changed(before.get("position", {}), after.get("position", {}), eps)
            or cls._dict_changed(before.get("rotation", {}), after.get("rotation", {}), eps)
            or cls._value_changed(before.get("horizon"), after.get("horizon"), eps)
        )

    @classmethod
    def _dict_changed(cls, before: dict, after: dict, eps: float) -> bool:
        keys = set(before) | set(after)
        return any(cls._value_changed(before.get(key), after.get(key), eps) for key in keys)

    @staticmethod
    def _value_changed(before, after, eps: float) -> bool:
        if before is None and after is None:
            return False
        try:
            return abs(float(before) - float(after)) > eps
        except (TypeError, ValueError):
            return before != after

    def _interacted_objects_after_action(
        self,
        action: str,
        action_kwargs: dict,
        before_meta: dict,
        after_meta: dict,
    ) -> list[dict]:
        object_ids = self._interaction_object_ids(action, action_kwargs, before_meta, after_meta)
        objects = []
        for object_id in object_ids:
            after_state = self._object_state_from_metadata(after_meta, object_id)
            before_state = self._object_state_from_metadata(before_meta, object_id)
            inventory_state = self._inventory_object_state(after_meta, object_id)
            current_state = after_state or inventory_state or before_state
            if current_state:
                state = dict(current_state)
            else:
                state = {"id": object_id, "objectId": object_id, "missing": True}
            state["before"] = dict(before_state) if before_state else None
            state["after"] = dict(after_state or inventory_state) if (after_state or inventory_state) else None
            state["state_changed"] = before_state != (after_state or inventory_state)
            objects.append(state)
        return objects

    def _interaction_object_ids(
        self,
        action: str,
        action_kwargs: dict,
        before_meta: dict,
        after_meta: dict,
    ) -> list[str]:
        object_ids: list[str] = []

        def add(value):
            if isinstance(value, str) and value and value not in object_ids:
                object_ids.append(value)
            elif isinstance(value, list):
                for item in value:
                    add(item)

        for key, value in action_kwargs.items():
            key_lower = key.lower()
            if key_lower == "objectid" or key_lower.endswith("objectid") or key_lower.endswith("objectids"):
                add(value)

        before_inventory = [item.get("objectId") for item in before_meta.get("inventoryObjects", [])]
        after_inventory = [item.get("objectId") for item in after_meta.get("inventoryObjects", [])]
        action_lower = (action or "").lower()
        if action_lower in {"putobject", "drophandobject", "dropobject", "throwobject", "moveheldobject"}:
            add(before_inventory)
        if action_lower in {"pickupobject", "putobject", "drophandobject", "dropobject", "throwobject", "moveheldobject"}:
            add(after_inventory)
        if before_inventory != after_inventory:
            add(before_inventory)
            add(after_inventory)

        return object_ids

    @staticmethod
    def _object_state_from_metadata(meta: dict, object_id: str) -> Optional[dict]:
        for obj in meta.get("objects", []):
            if obj.get("objectId") == object_id:
                return {
                    "id": object_id,
                    "objectId": object_id,
                    "type": obj.get("objectType", object_id.split("|")[0]),
                    "name": obj.get("name", ""),
                    "position": obj.get("position", {}),
                    "rotation": obj.get("rotation", {}),
                    "distance": obj.get("distance", -1),
                    "visible": obj.get("visible", False),
                    "pickupable": obj.get("pickupable", False),
                    "moveable": obj.get("moveable", False),
                    "receptacle": obj.get("receptacle", False),
                    "openable": obj.get("openable", False),
                    "isOpen": obj.get("isOpen", False),
                    "toggleable": obj.get("toggleable", False),
                    "isToggled": obj.get("isToggled", False),
                    "sliceable": obj.get("sliceable", False),
                    "isSliced": obj.get("isSliced", False),
                    "dirtyable": obj.get("dirtyable", False),
                    "isDirty": obj.get("isDirty", False),
                    "breakable": obj.get("breakable", False),
                    "isBroken": obj.get("isBroken", False),
                    "cookable": obj.get("cookable", False),
                    "isCooked": obj.get("isCooked", False),
                    "canFillWithLiquid": obj.get("canFillWithLiquid", False),
                    "isFilledWithLiquid": obj.get("isFilledWithLiquid", False),
                    "fillLiquid": obj.get("fillLiquid"),
                    "isPickedUp": obj.get("isPickedUp", False),
                    "parentReceptacles": obj.get("parentReceptacles"),
                    "receptacleObjectIds": obj.get("receptacleObjectIds", []),
                }
        return None

    @staticmethod
    def _inventory_object_state(meta: dict, object_id: str) -> Optional[dict]:
        for item in meta.get("inventoryObjects", []):
            if item.get("objectId") == object_id:
                return {
                    "id": object_id,
                    "objectId": object_id,
                    "type": item.get("objectType", object_id.split("|")[0]),
                    "inInventory": True,
                }
        return None

    def reachable_positions_response(self, robot_ref=None) -> dict:
        robot = self.resolve_robot(robot_ref)
        positions = self._get_reachable_positions(robot.robot_id)
        return {"status": "success", "robot_id": robot.robot_id, "positions": positions}

    def _get_interactable_poses(
        self,
        robot_ref,
        object_id: str,
        positions: list[dict[str, Any]],
        *,
        max_distance: float | None = None,
    ) -> list[dict[str, Any]]:
        robot = self.resolve_robot(robot_ref)
        action: dict[str, Any] = {
            "action": "GetInteractablePoses",
            "agentId": robot.robot_id,
            "objectId": object_id,
            "positions": positions,
            "standings": [True],
        }
        if max_distance is not None:
            action["maxDistance"] = max_distance
        try:
            event = self._controller_step(action)
            metadata = self._event_for_robot(event, robot.robot_id).metadata
        except Exception as exc:
            raise NavigationPlanningError(
                "interactable_pose_query_failed",
                f"GetInteractablePoses failed: {exc}",
            ) from exc
        if not metadata.get("lastActionSuccess", False):
            raise NavigationPlanningError(
                "interactable_pose_query_failed",
                str(metadata.get("errorMessage") or "GetInteractablePoses failed"),
            )

        poses: list[dict[str, Any]] = []
        for raw_pose in metadata.get("actionReturn") or []:
            if not isinstance(raw_pose, dict) or raw_pose.get("standing") is False:
                continue
            try:
                position = _position_dict(raw_pose, field="interactable_pose")
                rotation = float(raw_pose.get("rotation", 0.0)) % 360.0
                horizon = float(raw_pose.get("horizon", 0.0))
            except (TypeError, ValueError, NavigationPlanningError):
                continue
            poses.append(
                {
                    "x": position["x"],
                    "y": position["y"],
                    "z": position["z"],
                    "rotation": rotation,
                    "horizon": horizon,
                    "standing": True,
                }
            )
        return poses

    @staticmethod
    def _interactable_pose_key(pose: dict[str, Any]) -> tuple[float, float, float, float]:
        return (
            round(float(pose.get("x", 0.0)), 3),
            round(float(pose.get("z", 0.0)), 3),
            round(float(pose.get("rotation", 0.0)) % 360.0, 3),
            round(float(pose.get("horizon", 0.0)), 3),
        )

    @staticmethod
    def _target_visible_in_state(state: dict[str, Any], object_id: str | None) -> bool:
        if not isinstance(object_id, str) or not object_id:
            return False
        objects = state.get("objects", []) if isinstance(state, dict) else []
        return any(
            isinstance(item, dict)
            and (item.get("id") == object_id or item.get("objectId") == object_id)
            and bool(item.get("visible"))
            for item in objects
        )

    def plan_goto(
        self,
        payload: dict,
        dynamic_obstacles: list[dict[str, Any]] | None = None,
        *,
        blocked_nodes: set[tuple[float, float]] | None = None,
        blocked_edges: set[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        excluded_target_poses: set[tuple[float, float, float, float]] | None = None,
    ) -> dict:
        if not isinstance(payload, dict):
            raise NavigationPlanningError("invalid_target", "payload must be an object")
        robot_ref = payload.get("robot_id", payload.get("robot", payload.get("agent_id")))
        robot = self.resolve_robot(robot_ref)
        state = self.capture_state(robot.robot_id, render_image=False)
        inventory = state.get("inventory") if isinstance(state.get("inventory"), list) else []
        robot_position = _position_dict(robot.position, field="robot.position")
        target_position, target = find_object_target(payload, state, robot_position)
        reachable = self._get_reachable_positions(robot.robot_id)
        if not reachable:
            raise NavigationPlanningError("no_reachable_positions", "GetReachablePositions returned no usable positions")

        has_object_target = target.get("kind") == "object"
        default_min_distance = DEFAULT_OBJECT_MIN_DISTANCE if has_object_target else 0.0
        if has_object_target and inventory:
            try:
                default_min_distance = max(
                    default_min_distance,
                    float(payload.get("held_object_min_distance", DEFAULT_HELD_OBJECT_MIN_DISTANCE)),
                )
            except (TypeError, ValueError):
                default_min_distance = max(default_min_distance, DEFAULT_HELD_OBJECT_MIN_DISTANCE)
        min_distance = payload.get("min_distance", default_min_distance)
        max_distance = payload.get("max_distance")
        grid_size = float(payload.get("grid_size", DEFAULT_GRID_SIZE))
        epsilon = float(payload.get("grid_epsilon", DEFAULT_GRID_EPSILON))
        rotate_step = float(payload.get("rotate_step_degrees", DEFAULT_ROTATE_STEP_DEGREES))
        max_actions = payload.get("max_actions")
        face_target = bool(payload.get("face_target", has_object_target))
        require_target_visible = has_object_target and bool(payload.get("require_target_visible", True))
        try:
            min_distance = float(min_distance)
            max_distance = float(max_distance) if max_distance is not None else None
            max_actions = int(max_actions) if max_actions is not None else None
        except (TypeError, ValueError) as exc:
            raise NavigationPlanningError("invalid_target", "navigation numeric options are invalid") from exc

        start_position = find_nearest_position(reachable, robot_position)
        if start_position is None:
            raise NavigationPlanningError("no_reachable_positions", "could not align robot pose to a reachable position")

        graph, node_positions = build_reachable_graph(reachable, grid_size=grid_size, epsilon=epsilon)
        start_node = _node_for_position(start_position)
        dynamic_obstacle_radius_default = 0.35 if dynamic_obstacles else 0.0
        try:
            dynamic_obstacle_radius = float(payload.get("dynamic_obstacle_radius", dynamic_obstacle_radius_default))
        except (TypeError, ValueError) as exc:
            raise NavigationPlanningError("invalid_target", "navigation numeric options are invalid") from exc
        blocked_nodes = set(blocked_nodes or set())
        blocked_edges = {_canonical_edge(start, end) for start, end in (blocked_edges or set())}
        dynamic_obstacle_summaries: list[dict[str, Any]] = []
        for obstacle in dynamic_obstacles or []:
            obstacle_position = obstacle.get("position") if isinstance(obstacle, dict) else None
            if not isinstance(obstacle_position, dict):
                continue
            try:
                clean_position = _position_dict(obstacle_position, field="obstacle.position")
            except NavigationPlanningError:
                continue
            obstacle_nodes = nodes_within_radius(
                node_positions,
                clean_position,
                radius=max(0.0, dynamic_obstacle_radius),
            )
            nearest_position = find_nearest_position(list(node_positions.values()), clean_position)
            summary = {
                "robot_id": obstacle.get("robot_id"),
                "name": obstacle.get("name"),
                "position": clean_position,
                "blocked_node_count": len(obstacle_nodes),
            }
            if nearest_position is not None:
                summary["nearest_node"] = {"x": nearest_position["x"], "z": nearest_position["z"]}
            dynamic_obstacle_summaries.append(summary)
            blocked_nodes.update(obstacle_nodes)

        planning_reachable = [
            position
            for position in reachable
            if _node_for_position(position) == start_node or _node_for_position(position) not in blocked_nodes
        ]
        excluded_target_poses = set(excluded_target_poses or set())
        candidate_specs: list[dict[str, Any]] = []
        interactable_pose_count = 0
        if require_target_visible:
            target_object_id = target.get("object_id")
            if not isinstance(target_object_id, str) or not target_object_id:
                raise NavigationPlanningError(
                    "target_not_found",
                    "object target did not include an objectId",
                )
            interactable_poses = self._get_interactable_poses(
                robot.robot_id,
                target_object_id,
                planning_reachable,
                max_distance=max_distance,
            )
            interactable_pose_count = len(interactable_poses)
            for pose in interactable_poses:
                pose_key = self._interactable_pose_key(pose)
                goal_node = _node_for_position(pose)
                candidate = node_positions.get(goal_node)
                if candidate is None or pose_key in excluded_target_poses:
                    continue
                distance = _distance_xz(candidate, target_position)
                if distance < max(0.0, min_distance):
                    continue
                if max_distance is not None and distance > max_distance:
                    continue
                candidate_specs.append({"position": candidate, "pose": pose})
            if not candidate_specs:
                raise NavigationPlanningError(
                    "no_interactable_pose",
                    "no unblocked interactable pose satisfies the target distance constraints",
                )
        else:
            goal_candidates = choose_reachable_goal_candidates(
                planning_reachable,
                target_position,
                min_distance=max(0.0, min_distance),
                max_distance=max_distance,
            )
            if not goal_candidates:
                raise NavigationPlanningError(
                    "no_reachable_goal_near_target",
                    "no reachable position satisfies the target distance constraints",
                )
            candidate_specs = [{"position": candidate, "pose": None} for candidate in goal_candidates]

        graph = remove_graph_nodes(graph, blocked_nodes, preserve_nodes={start_node})
        graph = remove_graph_edges(graph, blocked_edges)
        current_yaw = float((robot.rotation or {}).get("y", 0.0))
        current_horizon = float(robot.horizon or 0.0)
        best_plan: dict[str, Any] | None = None
        best_score: tuple[Any, ...] | None = None
        reachable_goal_count = 0
        action_limited_goal_count = 0
        for candidate_spec in candidate_specs:
            candidate = candidate_spec["position"]
            goal_pose = candidate_spec.get("pose")
            goal_node = _node_for_position(candidate)
            node_path = astar_path(graph, start_node, goal_node)
            if node_path is None:
                continue
            reachable_goal_count += 1
            candidate_path = [node_positions[node] for node in node_path]
            candidate_actions = path_to_actions(
                candidate_path,
                current_yaw=current_yaw,
                rotate_step_degrees=rotate_step,
            )
            final_yaw = current_yaw if len(candidate_path) < 2 else _yaw_for_step(
                candidate_path[-2],
                candidate_path[-1],
            )
            face_target_yaw = (
                float(goal_pose["rotation"])
                if isinstance(goal_pose, dict)
                else _yaw_for_step(candidate, target_position)
            )
            if face_target or isinstance(goal_pose, dict):
                candidate_actions.extend(
                    yaw_rotation_actions(
                        current_yaw=final_yaw,
                        target_yaw=face_target_yaw,
                        rotate_step_degrees=rotate_step,
                    )
                )
            goal_horizon = float(goal_pose["horizon"]) if isinstance(goal_pose, dict) else current_horizon
            if isinstance(goal_pose, dict):
                candidate_actions.extend(
                    horizon_rotation_actions(
                        current_horizon=current_horizon,
                        target_horizon=goal_horizon,
                    )
                )
            if max_actions is not None and len(candidate_actions) > max_actions:
                action_limited_goal_count += 1
                continue
            estimated_distance = sum(
                _distance_xz(current, next_position)
                for current, next_position in zip(candidate_path, candidate_path[1:])
            )
            candidate_plan = {
                "goal_position": candidate,
                "goal_pose": dict(goal_pose) if isinstance(goal_pose, dict) else None,
                "goal_rotation": face_target_yaw,
                "goal_horizon": goal_horizon,
                "path": candidate_path,
                "actions": candidate_actions,
                "estimated_distance": estimated_distance,
                "face_target_yaw": face_target_yaw,
            }
            if not require_target_visible:
                best_plan = candidate_plan
                break
            candidate_score = (
                estimated_distance,
                len(candidate_actions),
                _distance_xz(candidate, target_position),
                self._interactable_pose_key(goal_pose),
            )
            if best_score is None or candidate_score < best_score:
                best_plan = candidate_plan
                best_score = candidate_score
        if best_plan is None:
            if reachable_goal_count == 0:
                raise NavigationPlanningError("no_path", "no path between start and any target-near goal")
            raise NavigationPlanningError(
                "action_limit_exceeded",
                f"all {action_limited_goal_count} reachable target-near goals exceed max_actions={max_actions}",
            )
        goal_position = best_plan["goal_position"]
        path = best_plan["path"]
        actions = best_plan["actions"]
        estimated_distance = best_plan["estimated_distance"]
        face_target_yaw = best_plan["face_target_yaw"]
        return {
            "status": "success",
            "robot_id": robot.robot_id,
            "target": target,
            "target_position": target_position,
            "start_position": start_position,
            "goal_position": goal_position,
            "path": path,
            "actions": actions,
            "estimated_distance": estimated_distance,
            "face_target": face_target,
            "face_target_yaw": face_target_yaw,
            "require_target_visible": require_target_visible,
            "target_pose_source": (
                "ai2thor_get_interactable_poses" if require_target_visible else None
            ),
            "interactable_pose_count": interactable_pose_count,
            "excluded_target_pose_count": len(excluded_target_poses),
            "goal_rotation": best_plan["goal_rotation"],
            "goal_horizon": best_plan["goal_horizon"],
            "goal_pose": best_plan["goal_pose"],
            "planner": "reachable_positions_astar",
            "grid_size": grid_size,
            "rotate_step_degrees": rotate_step,
            "holding_object": bool(inventory),
            "dynamic_obstacles": dynamic_obstacle_summaries,
            "blocked_node_count": len(blocked_nodes),
            "blocked_edge_count": len(blocked_edges),
            "blocked_edges": [serialize_graph_edge(edge) for edge in sorted(blocked_edges)],
            "goal_candidate_count": len(candidate_specs),
            "reachable_goal_candidate_count": reachable_goal_count,
        }

    def _dynamic_obstacles_for_robot(self, robot_id: int) -> list[dict[str, Any]]:
        obstacles: list[dict[str, Any]] = []
        for robot in self.robots:
            if robot.robot_id == robot_id:
                continue
            if not isinstance(robot.position, dict):
                continue
            obstacles.append({
                "robot_id": robot.robot_id,
                "name": robot.name,
                "position": dict(robot.position),
            })
        return obstacles

    @staticmethod
    def _first_failed_action(execute_result: dict) -> dict | None:
        for item in execute_result.get("results", []):
            if not item.get("success"):
                return item
        return None

    @staticmethod
    def _execution_status(results: list[dict[str, Any]], expected_count: int) -> str:
        if len(results) == expected_count and all(item.get("success") for item in results):
            return "success"
        if any(item.get("success") for item in results):
            return "partial"
        return "failed"

    def _object_obstacle_from_failure(
        self,
        blocking_object_id: str | None,
        *,
        executor_robot_id: int,
    ) -> dict[str, Any] | None:
        if not blocking_object_id:
            return None
        state = self._safe_capture_state(executor_robot_id)
        if not isinstance(state, dict):
            return None
        inventory_ids = {
            str(item.get("objectId") or item.get("id"))
            for item in state.get("inventory") or []
            if isinstance(item, dict)
        }
        if blocking_object_id in inventory_ids:
            return None
        objects = state.get("objects") if isinstance(state.get("objects"), list) else []
        type_hint = blocking_object_id.split("|")[0].split("_")[0]
        matches: list[dict[str, Any]] = []
        for obj in objects:
            if not isinstance(obj, dict):
                continue
            object_id = str(obj.get("objectId") or obj.get("id") or "")
            object_type = str(obj.get("objectType") or obj.get("type") or object_id.split("|")[0])
            object_name = str(obj.get("name") or "")
            if (
                object_id == blocking_object_id
                or object_name == blocking_object_id
                or object_id.startswith(blocking_object_id)
                or blocking_object_id.startswith(object_type)
                or object_type == type_hint
            ):
                matches.append(obj)
        if not matches:
            return None
        executor = self.resolve_robot(executor_robot_id)
        executor_position = executor.position if isinstance(executor.position, dict) else None
        if isinstance(executor_position, dict):
            matches.sort(
                key=lambda obj: _distance_xz(obj.get("position", {}), executor_position)
                if isinstance(obj.get("position"), dict)
                else float("inf")
            )
        obj = matches[0]
        position = obj.get("position") if isinstance(obj.get("position"), dict) else None
        if not isinstance(position, dict):
            return None
        return {
            "object_id": str(obj.get("objectId") or obj.get("id") or blocking_object_id),
            "object_type": str(obj.get("objectType") or obj.get("type") or type_hint),
            "name": str(obj.get("name") or blocking_object_id),
            "position": dict(position),
            "source": "navigation_failure_object_blocker",
        }

    def _navigation_failure_memory(
        self,
        failed_action: dict[str, Any] | None,
        plan: dict[str, Any],
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        blocker = parse_navigation_blocker(failed_action)
        try:
            grid_size = float(payload.get("grid_size", plan.get("grid_size", DEFAULT_GRID_SIZE)))
        except (TypeError, ValueError):
            grid_size = DEFAULT_GRID_SIZE
        edge = failed_movement_edge(failed_action, grid_size=grid_size)
        blocked_edges = [edge] if edge else []
        blocked_nodes: list[tuple[float, float]] = []
        if edge and blocker.get("failure_class") != "agent_blocker":
            blocked_nodes.append(edge[1])
        learned_obstacles: list[dict[str, Any]] = []
        target = plan.get("target") if isinstance(plan.get("target"), dict) else {}
        blocking_object_id = str(blocker.get("blocking_object_id") or "")
        target_object_id = str(target.get("object_id") or "")
        target_object_type = str(target.get("object_type") or "")
        target_is_blocker = bool(
            blocking_object_id
            and (
                blocking_object_id == target_object_id
                or target_object_id.startswith(blocking_object_id)
                or blocking_object_id.startswith(target_object_type)
            )
        )
        if target_is_blocker:
            blocker["target_object_blocker"] = True
        if blocker.get("failure_class") == "object_blocker" and not target_is_blocker:
            executor_robot_id = int(plan.get("robot_id", payload.get("robot_id", 0)))
            obstacle = self._object_obstacle_from_failure(
                blocker.get("blocking_object_id"),
                executor_robot_id=executor_robot_id,
            )
            if obstacle is not None:
                learned_obstacles.append(obstacle)
        return {
            "failure": blocker,
            "blocked_edges": blocked_edges,
            "blocked_nodes": blocked_nodes,
            "learned_dynamic_obstacles": learned_obstacles,
            "serialized_blocked_edges": [serialize_graph_edge(item) for item in blocked_edges],
            "serialized_blocked_nodes": [serialize_graph_node(item) for item in blocked_nodes],
        }

    def _choose_yield_parking_position(
        self,
        *,
        blocking_robot_id: int,
        executor_robot_id: int,
        avoid_positions: list[dict[str, Any]],
        min_clearance: float,
        max_distance: float,
    ) -> dict[str, float] | None:
        try:
            blocking_robot = self.resolve_robot(blocking_robot_id)
        except ValueError:
            return None
        start_position = blocking_robot.position if isinstance(blocking_robot.position, dict) else None
        if not isinstance(start_position, dict):
            return None
        reachable = self._get_reachable_positions(blocking_robot_id)
        if not reachable:
            return None

        def safe_enough(position: dict[str, Any], *, enforce_max_distance: bool) -> bool:
            try:
                if _distance_xz(position, start_position) < DEFAULT_GRID_SIZE * 0.5:
                    return False
                if enforce_max_distance and _distance_xz(position, start_position) > max_distance:
                    return False
                return all(_distance_xz(position, avoid) >= min_clearance for avoid in avoid_positions if isinstance(avoid, dict))
            except (KeyError, TypeError, ValueError):
                return False

        candidates = [pos for pos in reachable if safe_enough(pos, enforce_max_distance=True)]
        if not candidates:
            candidates = [pos for pos in reachable if safe_enough(pos, enforce_max_distance=False)]
        if not candidates:
            return None
        return dict(min(candidates, key=lambda pos: _distance_xz(pos, start_position)))

    def _attempt_blocking_robot_yield(
        self,
        *,
        blocker: dict[str, Any],
        executor_plan: dict[str, Any],
        payload: dict[str, Any],
        render_image: bool,
    ) -> dict[str, Any]:
        if blocker.get("failure_class") != "agent_blocker":
            return {"status": "skipped", "reason": "failure is not an agent blocker"}
        blocking_robot_id = blocker.get("blocking_robot_id")
        executor_robot_id = executor_plan.get("robot_id")
        if blocking_robot_id is None or executor_robot_id is None:
            return {"status": "skipped", "reason": "missing robot ids"}
        try:
            blocking_robot_id = int(blocking_robot_id)
            executor_robot_id = int(executor_robot_id)
        except (TypeError, ValueError):
            return {"status": "skipped", "reason": "invalid robot ids"}
        if blocking_robot_id == executor_robot_id:
            return {"status": "skipped", "reason": "blocking robot is executor"}

        executor = self.resolve_robot(executor_robot_id)
        avoid_positions = []
        if isinstance(executor.position, dict):
            avoid_positions.append(dict(executor.position))
        for key in ("start_position", "goal_position", "target_position"):
            value = executor_plan.get(key)
            if isinstance(value, dict):
                avoid_positions.append(dict(value))
        path = executor_plan.get("path") if isinstance(executor_plan.get("path"), list) else []
        if len(path) > 1 and isinstance(path[1], dict):
            avoid_positions.append(dict(path[1]))

        try:
            min_clearance = float(payload.get("yield_min_clearance", 0.65))
            max_distance = float(payload.get("yield_max_distance", 1.5))
            max_actions = int(payload.get("yield_max_actions", 12))
        except (TypeError, ValueError):
            return {"status": "failed", "error_code": "invalid_yield_options", "error": "yield options are invalid"}
        parking_position = self._choose_yield_parking_position(
            blocking_robot_id=blocking_robot_id,
            executor_robot_id=executor_robot_id,
            avoid_positions=avoid_positions,
            min_clearance=max(0.0, min_clearance),
            max_distance=max(DEFAULT_GRID_SIZE, max_distance),
        )
        trace: dict[str, Any] = {
            "blocking_robot_id": blocking_robot_id,
            "blocked_robot_id": executor_robot_id,
            "avoid_position_count": len(avoid_positions),
            "parking_position": parking_position,
        }
        if parking_position is None:
            trace.update({"status": "failed", "error_code": "blocked_by_agent_no_safe_yield", "error": "no safe parking point for blocking robot"})
            return trace

        yield_payload = {
            "task_id": f"{payload.get('task_id', 'goto')}_yield_robot_{blocking_robot_id}",
            "robot_id": blocking_robot_id,
            "target_position": parking_position,
            "min_distance": 0.0,
            "max_distance": 0.0,
            "face_target": False,
            "max_actions": max_actions,
            "grid_size": payload.get("grid_size", DEFAULT_GRID_SIZE),
            "grid_epsilon": payload.get("grid_epsilon", DEFAULT_GRID_EPSILON),
            "rotate_step_degrees": payload.get("rotate_step_degrees", DEFAULT_ROTATE_STEP_DEGREES),
        }
        executor_obstacle = {
            "robot_id": executor_robot_id,
            "name": getattr(executor, "name", f"Robot{executor_robot_id}"),
            "position": dict(executor.position),
        } if isinstance(executor.position, dict) else None
        try:
            yield_plan = self.plan_goto(
                yield_payload,
                dynamic_obstacles=[executor_obstacle] if executor_obstacle else [],
            )
            yield_actions = held_object_safe_navigation_actions(
                list(yield_plan.get("actions", [])),
                holding_object=bool(yield_plan.get("holding_object")),
                default_rotate_degrees=float(yield_plan.get("rotate_step_degrees", DEFAULT_ROTATE_STEP_DEGREES)),
                max_rotate_degrees=float(payload.get("held_max_rotate_degrees", 15.0)),
            )
        except (TypeError, ValueError, NavigationPlanningError) as exc:
            trace.update({
                "status": "failed",
                "error_code": getattr(exc, "error_code", "yield_planning_failed"),
                "error": str(exc),
            })
            return trace
        trace.update({
            "yield_plan_path_length": len(yield_plan.get("path", [])),
            "yield_action_count": len(yield_actions),
        })
        execute_result = self.execute_batch(
            yield_actions,
            default_robot_ref=blocking_robot_id,
            render_image=render_image,
            stop_on_failure=True,
        )
        trace["execute_result"] = execute_result
        trace["status"] = "success" if execute_result.get("status") == "success" else "failed"
        if trace["status"] != "success":
            trace["error_code"] = "blocking_robot_yield_failed"
            failed = self._first_failed_action(execute_result)
            if failed is not None:
                trace["failed_action"] = failed
                trace["error"] = failed.get("error")
        return trace

    def _plan_goto_with_dynamic_obstacles(
        self,
        payload: dict,
        *,
        avoid_other_robots: bool,
        learned_dynamic_obstacles: list[dict[str, Any]] | None = None,
        blocked_nodes: set[tuple[float, float]] | None = None,
        blocked_edges: set[tuple[tuple[float, float], tuple[float, float]]] | None = None,
        excluded_target_poses: set[tuple[float, float, float, float]] | None = None,
    ) -> dict:
        robot_ref = payload.get("robot_id", payload.get("robot", payload.get("agent_id")))
        robot = self.resolve_robot(robot_ref)
        dynamic_obstacles = self._dynamic_obstacles_for_robot(robot.robot_id) if avoid_other_robots else []
        dynamic_obstacles.extend(learned_dynamic_obstacles or [])
        return self.plan_goto(
            payload,
            dynamic_obstacles=dynamic_obstacles,
            blocked_nodes=blocked_nodes,
            blocked_edges=blocked_edges,
            excluded_target_poses=excluded_target_poses,
        )

    def goto(self, payload: dict) -> dict:
        task_id = payload.get("task_id", "unknown") if isinstance(payload, dict) else "unknown"
        execute = bool(payload.get("execute", False)) if isinstance(payload, dict) else False
        avoid_other_robots = bool(payload.get("avoid_other_robots", True)) if isinstance(payload, dict) else True
        robot_ref = payload.get("robot_id", payload.get("robot", payload.get("agent_id"))) if isinstance(payload, dict) else None
        learned_blocked_edges: set[tuple[tuple[float, float], tuple[float, float]]] = set()
        learned_blocked_nodes: set[tuple[float, float]] = set()
        learned_dynamic_obstacles: list[dict[str, Any]] = []
        learned_dynamic_obstacle_keys: set[tuple[Any, ...]] = set()
        excluded_target_poses: set[tuple[float, float, float, float]] = set()
        negotiation_trace: list[dict[str, Any]] = []
        last_failure_class: str | None = None
        try:
            if execute:
                plan = self._plan_goto_with_dynamic_obstacles(
                    payload,
                    avoid_other_robots=avoid_other_robots,
                    learned_dynamic_obstacles=learned_dynamic_obstacles,
                    blocked_nodes=learned_blocked_nodes,
                    blocked_edges=learned_blocked_edges,
                    excluded_target_poses=excluded_target_poses,
                )
            else:
                plan = self.plan_goto(payload)
        except NavigationPlanningError as exc:
            return {"status": "failed", "task_id": task_id, "error_code": exc.error_code, "error": str(exc)}
        result = {"task_id": task_id, **plan, "execute": execute}
        if not execute:
            return result

        render_image = bool(payload.get("render_image", False))
        stop_on_failure = bool(payload.get("stop_on_failure", True))
        try:
            max_replans = max(0, int(payload.get("max_replans", 3)))
            chunk_size = max(1, int(payload.get("replan_chunk_actions", 6)))
        except (TypeError, ValueError):
            result.update({
                "status": "failed",
                "error_code": "invalid_target",
                "error": "navigation replanning options are invalid",
            })
            return result

        all_results: list[dict[str, Any]] = []
        replan_trace: list[dict[str, Any]] = []
        replan_count = 0
        plan_attempt = 0
        last_failure: dict | None = None
        expected_action_count = 0

        while True:
            trace_item: dict[str, Any] = {
                "attempt": plan_attempt,
                "replan_count": replan_count,
                "start_position": plan.get("start_position"),
                "goal_position": plan.get("goal_position"),
                "path_length": len(plan.get("path", [])),
                "action_count": len(plan.get("actions", [])),
                "dynamic_obstacles": plan.get("dynamic_obstacles", []),
                "blocked_node_count": plan.get("blocked_node_count", 0),
                "blocked_edge_count": plan.get("blocked_edge_count", 0),
                "learned_blocked_node_count": len(learned_blocked_nodes),
                "learned_blocked_edge_count": len(learned_blocked_edges),
                "learned_dynamic_obstacle_count": len(learned_dynamic_obstacles),
                "segments": [],
            }
            replan_trace.append(trace_item)
            try:
                held_max_rotate_degrees = float(
                    payload.get("held_max_rotate_degrees", 15.0)
                )
                actions = held_object_safe_navigation_actions(
                    list(plan.get("actions", [])),
                    holding_object=bool(plan.get("holding_object")),
                    default_rotate_degrees=float(
                        plan.get("rotate_step_degrees", DEFAULT_ROTATE_STEP_DEGREES)
                    ),
                    max_rotate_degrees=held_max_rotate_degrees,
                )
            except (TypeError, ValueError, NavigationPlanningError) as exc:
                result.update({
                    "status": "failed",
                    "error_code": "invalid_target",
                    "error": str(exc),
                })
                return result
            trace_item["planned_action_count"] = len(plan.get("actions", []))
            trace_item["action_count"] = len(actions)
            trace_item["holding_object_rotation_safety"] = bool(plan.get("holding_object"))
            expected_action_count += len(actions)
            segment_failed = False
            for offset in range(0, len(actions), chunk_size):
                segment = actions[offset: offset + chunk_size]
                execute_result = self.execute_batch(
                    segment,
                    default_robot_ref=plan["robot_id"],
                    render_image=render_image,
                    stop_on_failure=stop_on_failure,
                )
                segment_summary = {
                    "action_offset": offset,
                    "action_count": len(segment),
                    "status": execute_result.get("status"),
                }
                failed_action = self._first_failed_action(execute_result)
                if failed_action is not None:
                    segment_summary["failed_action"] = failed_action
                trace_item["segments"].append(segment_summary)
                for item in execute_result.get("results", []):
                    item = dict(item)
                    item["index"] = len(all_results)
                    all_results.append(item)
                if execute_result.get("status") != "success":
                    last_failure = failed_action
                    segment_failed = True
                    break
            if not segment_failed:
                post_state = self._safe_capture_state(plan.get("robot_id", robot_ref))
                target = plan.get("target") if isinstance(plan.get("target"), dict) else {}
                target_object_id = target.get("object_id")
                require_target_visible = bool(plan.get("require_target_visible"))
                post_target_visible = (
                    self._target_visible_in_state(post_state, target_object_id)
                    if isinstance(post_state, dict) and isinstance(target_object_id, str)
                    else False
                )
                trace_item["post_target_visible"] = post_target_visible
                trace_item["post_target_object_id"] = target_object_id
                if require_target_visible and not post_target_visible:
                    trace_item["status"] = "target_not_visible_after_goto"
                    goal_pose = plan.get("goal_pose")
                    if isinstance(goal_pose, dict):
                        excluded_target_poses.add(self._interactable_pose_key(goal_pose))
                    trace_item["excluded_target_pose_count"] = len(excluded_target_poses)
                    last_failure_class = "target_visibility"
                    if replan_count >= max_replans:
                        result.update(plan)
                        result.update({
                            "status": "failed",
                            "error_code": "target_not_visible_after_goto",
                            "error": "target was not visible after all allowed interactable-pose attempts",
                            "post_target_visible": False,
                            "post_target_object_id": target_object_id,
                            "execute_result": {
                                "status": self._execution_status(all_results, expected_action_count),
                                "results": all_results,
                                "state": post_state,
                            },
                            "replan_count": replan_count,
                            "replan_trace": replan_trace,
                            "negotiation_trace": negotiation_trace,
                            "executed_action_count": len(all_results),
                            "avoid_other_robots": avoid_other_robots,
                            "failure_class": last_failure_class,
                        })
                        return result
                    replan_count += 1
                    plan_attempt += 1
                    try:
                        plan = self._plan_goto_with_dynamic_obstacles(
                            payload,
                            avoid_other_robots=avoid_other_robots,
                            learned_dynamic_obstacles=learned_dynamic_obstacles,
                            blocked_nodes=learned_blocked_nodes,
                            blocked_edges=learned_blocked_edges,
                            excluded_target_poses=excluded_target_poses,
                        )
                    except NavigationPlanningError as exc:
                        result.update(plan)
                        result.update({
                            "status": "failed",
                            "error_code": "target_not_visible_after_goto",
                            "error": "target remained invisible and no further interactable pose was available",
                            "replan_error_code": exc.error_code,
                            "replan_error": str(exc),
                            "post_target_visible": False,
                            "post_target_object_id": target_object_id,
                            "execute_result": {
                                "status": self._execution_status(all_results, expected_action_count),
                                "results": all_results,
                                "state": post_state,
                            },
                            "replan_count": replan_count,
                            "replan_trace": replan_trace,
                            "negotiation_trace": negotiation_trace,
                            "executed_action_count": len(all_results),
                            "avoid_other_robots": avoid_other_robots,
                            "failure_class": last_failure_class,
                        })
                        return result
                    continue
                trace_item["status"] = "success"
                result.update(plan)
                result.update({
                    "status": "success",
                    "post_target_visible": post_target_visible,
                    "post_target_object_id": target_object_id,
                    "execute_result": {
                        "status": "success",
                        "results": all_results,
                        "state": post_state,
                    },
                    "replan_count": replan_count,
                    "replan_trace": replan_trace,
                    "negotiation_trace": negotiation_trace,
                    "executed_action_count": len(all_results),
                    "avoid_other_robots": avoid_other_robots,
                    "learned_blocked_edges": [serialize_graph_edge(edge) for edge in sorted(learned_blocked_edges)],
                    "learned_blocked_nodes": [serialize_graph_node(node) for node in sorted(learned_blocked_nodes)],
                    "learned_dynamic_obstacles": learned_dynamic_obstacles,
                    "failure_class": last_failure_class,
                })
                return result
            failure_memory = self._navigation_failure_memory(last_failure, plan, payload)
            last_failure_info = failure_memory.get("failure") if isinstance(failure_memory.get("failure"), dict) else {}
            last_failure_class = last_failure_info.get("failure_class")
            trace_item["failure_memory"] = {
                "failure": last_failure_info,
                "blocked_edges": failure_memory.get("serialized_blocked_edges", []),
                "blocked_nodes": failure_memory.get("serialized_blocked_nodes", []),
                "learned_dynamic_obstacles": failure_memory.get("learned_dynamic_obstacles", []),
            }
            learned_blocked_edges.update(failure_memory.get("blocked_edges", []))
            learned_blocked_nodes.update(failure_memory.get("blocked_nodes", []))
            if (
                last_failure_info.get("target_object_blocker")
                and plan.get("holding_object")
                and isinstance(payload, dict)
            ):
                payload = dict(payload)
                try:
                    current_min = float(payload.get("min_distance", 0.0))
                except (TypeError, ValueError):
                    current_min = 0.0
                payload["min_distance"] = max(
                    current_min,
                    float(payload.get("held_object_retry_min_distance", 1.25)),
                )
                trace_item["adjusted_min_distance_for_held_object"] = payload["min_distance"]
            for obstacle in failure_memory.get("learned_dynamic_obstacles", []):
                key = (
                    obstacle.get("source"),
                    obstacle.get("robot_id"),
                    obstacle.get("object_id"),
                    _node_for_position(obstacle.get("position")) if isinstance(obstacle.get("position"), dict) else None,
                )
                if key in learned_dynamic_obstacle_keys:
                    continue
                learned_dynamic_obstacle_keys.add(key)
                learned_dynamic_obstacles.append(obstacle)

            if last_failure_info.get("failure_class") == "agent_blocker":
                negotiation = self._attempt_blocking_robot_yield(
                    blocker=last_failure_info,
                    executor_plan=plan,
                    payload=payload,
                    render_image=render_image,
                )
                negotiation_trace.append(negotiation)
                trace_item["negotiation"] = negotiation
                for item in (negotiation.get("execute_result") or {}).get("results", []):
                    item = dict(item)
                    item["index"] = len(all_results)
                    item["negotiation"] = True
                    all_results.append(item)

            if replan_count >= max_replans:
                result.update({
                    "status": "failed",
                    "error_code": "execution_failed_after_replans",
                    "error": "navigation action execution failed after replanning",
                    "execute_result": {
                        "status": self._execution_status(all_results, expected_action_count),
                        "results": all_results,
                        "state": self._safe_capture_state(plan.get("robot_id", robot_ref)),
                    },
                    "replan_count": replan_count,
                    "replan_trace": replan_trace,
                    "negotiation_trace": negotiation_trace,
                    "executed_action_count": len(all_results),
                    "failed_action": last_failure,
                    "avoid_other_robots": avoid_other_robots,
                    "learned_blocked_edges": [serialize_graph_edge(edge) for edge in sorted(learned_blocked_edges)],
                    "learned_blocked_nodes": [serialize_graph_node(node) for node in sorted(learned_blocked_nodes)],
                    "learned_dynamic_obstacles": learned_dynamic_obstacles,
                    "failure_class": last_failure_class,
                    "blocking_robot_id": last_failure_info.get("blocking_robot_id"),
                    "blocked_robot_id": last_failure_info.get("blocked_robot_id"),
                    "blocking_object_id": last_failure_info.get("blocking_object_id"),
                })
                return result
            replan_count += 1
            plan_attempt += 1
            try:
                plan = self._plan_goto_with_dynamic_obstacles(
                    payload,
                    avoid_other_robots=avoid_other_robots,
                    learned_dynamic_obstacles=learned_dynamic_obstacles,
                    blocked_nodes=learned_blocked_nodes,
                    blocked_edges=learned_blocked_edges,
                    excluded_target_poses=excluded_target_poses,
                )
            except NavigationPlanningError as exc:
                result.update({
                    "status": "failed",
                    "error_code": "execution_failed_after_replans",
                    "error": "navigation action execution failed and replanning did not find a valid path",
                    "replan_error_code": exc.error_code,
                    "replan_error": str(exc),
                    "execute_result": {
                        "status": self._execution_status(all_results, expected_action_count),
                        "results": all_results,
                        "state": self._safe_capture_state(plan.get("robot_id", robot_ref)),
                    },
                    "replan_count": replan_count,
                    "replan_trace": replan_trace,
                    "negotiation_trace": negotiation_trace,
                    "executed_action_count": len(all_results),
                    "failed_action": last_failure,
                    "avoid_other_robots": avoid_other_robots,
                    "learned_blocked_edges": [serialize_graph_edge(edge) for edge in sorted(learned_blocked_edges)],
                    "learned_blocked_nodes": [serialize_graph_node(node) for node in sorted(learned_blocked_nodes)],
                    "learned_dynamic_obstacles": learned_dynamic_obstacles,
                    "failure_class": last_failure_class,
                })
                return result

    def _select_spawn_positions(self, count: int) -> list[dict]:
        reachable = self._get_reachable_positions()
        if not reachable:
            reachable = self._fallback_positions(count)
        if len(reachable) <= count:
            return [dict(pos) for pos in reachable[:count]]
        center = self._position_center(reachable)
        first = max(reachable, key=lambda pos: self._dist2(pos, center))
        selected = [first]
        remaining = [pos for pos in reachable if pos is not first]
        while len(selected) < count and remaining:
            best = max(remaining, key=lambda pos: min(self._dist2(pos, item) for item in selected))
            selected.append(best)
            remaining.remove(best)
        return [dict(pos) for pos in selected]

    def _get_reachable_positions(self, robot_ref=None) -> list[dict]:
        try:
            if robot_ref is None and not self.robots:
                robot_id = 0
            else:
                robot_id = self.resolve_robot(robot_ref).robot_id
            event = self._controller_step({"action": "GetReachablePositions", "agentId": robot_id})
            meta = self._event_for_robot(event, robot_id).metadata
            positions = meta.get("actionReturn") or []
        except Exception as exc:
            log_event("SIM OUT", f"WARNING GetReachablePositions failed: {exc}")
            return []
        cleaned = []
        for pos in positions:
            try:
                cleaned.append(_position_dict(pos))
            except NavigationPlanningError:
                continue
        return cleaned

    def _select_position_in_front_of_object(self, object_type: str, fallback: dict) -> dict:
        target_pos = self._find_object_position(object_type)
        if not target_pos:
            log_event("SIM OUT", f"WARNING cannot place Robot0 at {object_type}: object not found")
            return fallback
        reachable = self._get_reachable_positions()
        if not reachable:
            return fallback
        min_distance = max(0.1, self.robot0_fridge_distance)
        candidates = [pos for pos in reachable if self._dist2(pos, target_pos) >= min_distance * min_distance]
        if not candidates:
            candidates = reachable
        selected = min(candidates, key=lambda pos: self._dist2(pos, target_pos))
        log_event(
            "SIM OUT",
            f"Robot0 placed near {object_type}: object=({target_pos.get('x', 0):.2f}, "
            f"{target_pos.get('z', 0):.2f}), stand=({selected['x']:.2f}, {selected['z']:.2f})",
        )
        return dict(selected)

    def _find_object_position(self, object_type: str) -> Optional[dict]:
        event = self._controller_step({"action": "Pass", "agentId": 0, "renderImage": False})
        meta = self._event_for_robot(event, 0).metadata
        for obj in meta.get("objects", []):
            if obj.get("objectType") == object_type and obj.get("position"):
                return obj["position"]
        return None

    def _avoid_spawn_overlap(self, positions: list[dict], min_distance: float = 0.75) -> list[dict]:
        reachable = self._get_reachable_positions()
        if not reachable:
            return positions
        adjusted = [dict(positions[0])]
        min_dist2 = min_distance * min_distance
        for pos in positions[1:]:
            if all(self._dist2(pos, existing) >= min_dist2 for existing in adjusted):
                adjusted.append(dict(pos))
                continue
            replacement = max(reachable, key=lambda candidate: min(self._dist2(candidate, existing) for existing in adjusted))
            adjusted.append(dict(replacement))
            log_event(
                "SIM OUT",
                f"Adjusted spawn to avoid overlap: ({pos['x']:.2f}, {pos['z']:.2f}) -> "
                f"({replacement['x']:.2f}, {replacement['z']:.2f})",
            )
        return adjusted

    def _apply_robot0_initial_offset(self, pos: dict, yaw: float) -> tuple[dict, float]:
        if not any((self.robot0_dx, self.robot0_dz, self.robot0_left, self.robot0_right, self.robot0_back, self.robot0_dyaw)):
            return pos, yaw
        adjusted = dict(pos)
        adjusted["x"] += self.robot0_dx
        adjusted["z"] += self.robot0_dz
        yaw_rad = math.radians(yaw)
        lateral = self.robot0_left - self.robot0_right
        if lateral:
            adjusted["x"] += lateral * math.cos(yaw_rad)
            adjusted["z"] += lateral * math.sin(yaw_rad)
        if self.robot0_back:
            adjusted["x"] -= self.robot0_back * math.sin(yaw_rad)
            adjusted["z"] -= self.robot0_back * math.cos(yaw_rad)
        adjusted_yaw = yaw + self.robot0_dyaw
        log_event(
            "SIM IN",
            f"Apply Robot0 initial offset: dx={self.robot0_dx:.2f}, dz={self.robot0_dz:.2f}, "
            f"left={self.robot0_left:.2f}, right={self.robot0_right:.2f}, back={self.robot0_back:.2f}, "
            f"dyaw={self.robot0_dyaw:.1f}",
        )
        return adjusted, adjusted_yaw

    def initialize_windows(self):
        cv2 = get_cv2()
        if cv2 is None:
            raise RuntimeError(f"OpenCV import failed: {_CV2_IMPORT_ERROR}")
        self._window_names = {}
        for robot in self.robots:
            name = f"{robot.name} Native Controller View"
            self._window_names[robot.robot_id] = name
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, self.width, self.height)
            col = robot.robot_id % 3
            row = robot.robot_id // 3
            cv2.moveWindow(name, 40 + col * (self.width + 30), 60 + row * (self.height + 60))
        log_event("LOCAL OUT", f"Opened {len(self._window_names)} robot view windows")

    def display_loop(self, fps: float = 5.0):
        cv2 = get_cv2()
        if cv2 is None:
            log_event("LOCAL OUT", "OpenCV unavailable; running without local windows")
            log_event("LOCAL OUT", f"OpenCV import error: {_CV2_IMPORT_ERROR}")
            return False
        try:
            self.initialize_windows()
        except Exception as exc:
            log_event("LOCAL OUT", "OpenCV cannot create windows; running without local windows")
            log_event("LOCAL OUT", f"OpenCV error: {exc}")
            return False
        delay_ms = max(1, int(1000 / max(0.1, fps)))
        try:
            while True:
                try:
                    with self.lock:
                        event = self._controller_step({"action": "Pass", "agentId": 0, "renderImage": True})
                        self._refresh_all_robot_states(event)
                        events = self._events_from(event)
                        for robot in self.robots:
                            if robot.robot_id >= len(events):
                                continue
                            img = self._event_bgr_image(events[robot.robot_id])
                            if img is None:
                                continue
                            img = img.copy()
                            self._draw_overlay(img, robot)
                            cv2.imshow(self._window_names[robot.robot_id], img)
                except Exception as exc:
                    log_event(
                        "LOCAL OUT",
                        f"Display loop stopped after controller error: {type(exc).__name__}: {exc}",
                    )
                    traceback.print_exc()
                    return False
                key = cv2.waitKey(delay_ms) & 0xFF
                if key in (ord("q"), 27):
                    log_event("LOCAL IN", "GUI quit requested")
                    break
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        return True

    @staticmethod
    def _draw_overlay(img, robot: RobotState):
        cv2 = get_cv2()
        if cv2 is None:
            return
        pos = robot.position
        color = (0, 255, 0) if robot.last_success else (0, 0, 255)
        cv2.putText(img, f"{robot.name} | native controller | action #{robot.action_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, f"Pos: ({pos['x']:.2f}, {pos.get('y', DEFAULT_AGENT_Y):.2f}, {pos['z']:.2f})", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
        cv2.putText(img, f"{robot.last_action}: {'OK' if robot.last_success else 'FAIL'}", (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    @staticmethod
    def _objects_from_metadata(meta: dict) -> list[dict]:
        objects = []
        for obj in meta.get("objects", []):
            object_id = obj.get("objectId", "")
            if object_id.startswith("Floor|"):
                continue
            objects.append(
                {
                    "id": object_id,
                    "objectId": object_id,
                    "type": obj.get("objectType", object_id.split("|")[0]),
                    "objectType": obj.get("objectType", object_id.split("|")[0]),
                    "name": obj.get("name", ""),
                    "position": obj.get("position", {}),
                    "rotation": obj.get("rotation", {}),
                    "axisAlignedBoundingBox": obj.get("axisAlignedBoundingBox"),
                    "distance": obj.get("distance", -1),
                    "pickupable": obj.get("pickupable", False),
                    "moveable": obj.get("moveable", False),
                    "receptacle": obj.get("receptacle", False),
                    "openable": obj.get("openable", False),
                    "isOpen": obj.get("isOpen", False),
                    "toggleable": obj.get("toggleable", False),
                    "isToggled": obj.get("isToggled", False),
                    "sliceable": obj.get("sliceable", False),
                    "isSliced": obj.get("isSliced", False),
                    "dirtyable": obj.get("dirtyable", False),
                    "isDirty": obj.get("isDirty", False),
                    "breakable": obj.get("breakable", False),
                    "isBroken": obj.get("isBroken", False),
                    "cookable": obj.get("cookable", False),
                    "isCooked": obj.get("isCooked", False),
                    "canFillWithLiquid": obj.get("canFillWithLiquid", False),
                    "isFilledWithLiquid": obj.get("isFilledWithLiquid", False),
                    "fillLiquid": obj.get("fillLiquid"),
                    "isPickedUp": obj.get("isPickedUp", False),
                    "parentReceptacles": obj.get("parentReceptacles"),
                    "receptacleObjectIds": obj.get("receptacleObjectIds"),
                    "openness": obj.get("openness"),
                    "isMoving": obj.get("isMoving", False),
                    "visible": obj.get("visible", False),
                }
            )
        objects.sort(key=lambda item: item["distance"] if item["distance"] >= 0 else float("inf"))
        return objects

    @staticmethod
    def _event_bgr_image(event):
        if event is None:
            return None
        if hasattr(event, "cv2img") and event.cv2img is not None:
            return event.cv2img
        cv2 = get_cv2()
        if cv2 is not None and hasattr(event, "frame") and event.frame is not None:
            return event.frame[:, :, ::-1]
        return None

    @classmethod
    def _encode_image(cls, event) -> Optional[str]:
        if hasattr(event, "frame") and event.frame is not None:
            try:
                from PIL import Image
            except Exception:
                pass
            else:
                image = Image.fromarray(event.frame)
                buf = BytesIO()
                image.save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        cv2 = get_cv2()
        if cv2 is not None:
            img = cls._event_bgr_image(event)
            if img is not None:
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    return base64.b64encode(buf.tobytes()).decode("utf-8")
        return None

    @staticmethod
    def _fallback_positions(count: int) -> list[dict]:
        if count == 1:
            return [{"x": 0.0, "y": DEFAULT_AGENT_Y, "z": 0.0}]
        radius = 1.25
        return [
            {"x": radius * math.cos(2 * math.pi * i / count), "y": DEFAULT_AGENT_Y, "z": radius * math.sin(2 * math.pi * i / count)}
            for i in range(count)
        ]

    @staticmethod
    def _position_center(positions: list[dict]) -> dict:
        if not positions:
            return {"x": 0.0, "y": DEFAULT_AGENT_Y, "z": 0.0}
        return {
            "x": sum(pos["x"] for pos in positions) / len(positions),
            "y": sum(pos.get("y", DEFAULT_AGENT_Y) for pos in positions) / len(positions),
            "z": sum(pos["z"] for pos in positions) / len(positions),
        }

    @staticmethod
    def _dist2(a: dict, b: dict) -> float:
        dx = a["x"] - b["x"]
        dz = a["z"] - b["z"]
        return dx * dx + dz * dz

    @staticmethod
    def _yaw_to_target(pos: dict, target: dict) -> float:
        dx = target["x"] - pos["x"]
        dz = target["z"] - pos["z"]
        if abs(dx) < 1e-6 and abs(dz) < 1e-6:
            return 0.0
        return math.degrees(math.atan2(-dx, dz))


thor_instance: Optional[NativeControllerThorServer] = None


class NativeControllerReceiverHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log_event("HTTP ACCESS", f"{self.address_string()} - {fmt % args}")

    def _set_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        log_event("REMOTE IN", f"GET {self.path}")
        if parsed.path in {"/", "/health"}:
            self._send_json(200, {"status": "ok", "service": "ai2thor_receiver_server", "robots": len(thor_instance.robots) if thor_instance else 0})
        elif parsed.path == "/robots":
            if not self._controller_ready():
                return
            self._send_json(200, {"robots": [robot.to_dict() for robot in thor_instance.robots]})
        elif parsed.path == "/state":
            if not self._controller_ready():
                return
            robot_ref = self._query_value(query, "robot_id")
            render_image = self._query_bool(query, "render_image")
            try:
                self._send_json(200, thor_instance.capture_state(robot_ref, render_image=render_image))
            except ValueError as exc:
                self._send_json(400, {"status": "failed", "error": str(exc)})
        elif parsed.path == "/reachable_positions":
            if not self._controller_ready():
                return
            robot_ref = self._query_value(query, "robot_id")
            try:
                self._send_json(200, thor_instance.reachable_positions_response(robot_ref))
            except ValueError as exc:
                self._send_json(400, {"status": "failed", "error": str(exc)})
        else:
            self._send_json(404, {"status": "failed", "error": "not_found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        log_event("REMOTE IN", f"POST {parsed.path}")
        if parsed.path == "/execute_actions":
            self._handle_execute()
        elif parsed.path == "/observe":
            self._handle_observe()
        elif parsed.path == "/goto":
            self._handle_goto()
        elif parsed.path == "/reset":
            self._handle_reset()
        else:
            self._send_json(404, {"status": "failed", "error": "not_found"})

    def _controller_ready(self) -> bool:
        if thor_instance is None or thor_instance.controller is None:
            self._send_json(503, {"status": "failed", "error": "controller not started"})
            return False
        return True

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8")) if body else {}

    @staticmethod
    def _strict_jsonable(value: Any):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, dict):
            return {str(key): NativeControllerReceiverHandler._strict_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [NativeControllerReceiverHandler._strict_jsonable(item) for item in value]

        item = getattr(value, "item", None)
        if callable(item):
            try:
                item_value = item()
            except Exception:
                item_value = None
            else:
                if item_value is not value and isinstance(item_value, (str, bool, int, float, dict, list, tuple, type(None))):
                    return NativeControllerReceiverHandler._strict_jsonable(item_value)

        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            try:
                list_value = tolist()
            except Exception:
                list_value = None
            else:
                if list_value is not value and isinstance(list_value, (str, bool, int, float, dict, list, tuple, type(None))):
                    return NativeControllerReceiverHandler._strict_jsonable(list_value)

        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable in execute_actions result")

    @classmethod
    def _strict_json_body(cls, payload: dict) -> bytes:
        jsonable = cls._strict_jsonable(payload)
        return json.dumps(jsonable, ensure_ascii=False, allow_nan=False).encode("utf-8")

    def _send_preencoded_json(self, status_code: int, body: bytes):
        self.send_response(status_code)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _send_execute_result_json(self, status_code: int, task_id, result: dict) -> None:
        try:
            body = self._strict_json_body(result)
        except Exception as exc:
            log_event("REMOTE OUT", f"POST /execute_actions result serialization exception: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            error_payload = {
                "status": "failed",
                "task_id": task_id,
                "error_code": "execute_result_not_serializable",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            body = self._strict_json_body(error_payload)
            self._send_preencoded_json(500, body)
            return
        self._send_preencoded_json(status_code, body)

    def _handle_execute(self):
        if not self._controller_ready():
            return
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(400, {"status": "failed", "error": f"invalid json: {exc}"})
            return
        task_id = payload.get("task_id", "unknown")
        default_robot_ref = payload.get("robot_id", payload.get("robot", payload.get("agent_id")))
        actions = payload.get("actions", [])
        render_image = bool(payload.get("render_image", False))
        stop_on_failure = bool(payload.get("stop_on_failure", True))
        if not actions:
            self._send_json(400, {"status": "failed", "error": "empty actions"})
            return
        log_event("REMOTE IN", f"POST /execute_actions task_id={task_id}, default_robot={default_robot_ref}, actions={len(actions)}, render_image={render_image}", blank_before=True)
        for action in actions:
            log_event("REMOTE IN", f"action {json.dumps(action, ensure_ascii=False)}")
        try:
            result = thor_instance.execute_batch(actions, default_robot_ref=default_robot_ref, render_image=render_image, stop_on_failure=stop_on_failure)
            result["task_id"] = task_id
            results = result.get("results", []) if isinstance(result, dict) else []
            n_ok = sum(1 for item in results if isinstance(item, dict) and item.get("success"))
            log_event("REMOTE OUT", f"POST /execute_actions status={result.get('status')} ({n_ok}/{len(results)} ok)")
            for item in results:
                if isinstance(item, dict) and not item.get("success"):
                    log_event(
                        "REMOTE OUT",
                        "failed action "
                        f"index={item.get('index')} robot_id={item.get('robot_id')} "
                        f"action={item.get('action')} error={item.get('error')}",
                    )
        except Exception as exc:
            log_event("REMOTE OUT", f"POST /execute_actions receiver exception: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._send_json(
                500,
                {
                    "status": "failed",
                    "task_id": task_id,
                    "error_code": "receiver_exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return
        self._send_execute_result_json(200, task_id, result)

    def _handle_goto(self):
        if not self._controller_ready():
            return
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(400, {"status": "failed", "error": f"invalid json: {exc}"})
            return
        task_id = payload.get("task_id", "unknown")
        robot_ref = payload.get("robot_id", payload.get("robot", payload.get("agent_id")))
        execute = bool(payload.get("execute", False))
        log_event(
            "REMOTE IN",
            f"POST /goto task_id={task_id}, robot={robot_ref}, execute={execute}",
            blank_before=True,
        )
        try:
            result = thor_instance.goto(payload)
        except Exception as exc:
            log_event("REMOTE OUT", f"POST /goto receiver exception: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            self._send_json(
                500,
                {
                    "status": "failed",
                    "task_id": task_id,
                    "error_code": "receiver_exception",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return
        log_event(
            "REMOTE OUT",
            f"POST /goto status={result.get('status')} actions={len(result.get('actions', []))}",
        )
        status_code = 200 if result.get("status") == "success" else 400
        if result.get("error_code") == "execution_failed":
            status_code = 200
        self._send_json(status_code, result)

    def _handle_observe(self):
        if not self._controller_ready():
            return
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(400, {"status": "failed", "error": f"invalid json: {exc}"})
            return
        robot_ref = payload.get("robot_id", payload.get("robot", payload.get("agent_id")))
        render_image = bool(payload.get("render_image", True))
        try:
            self._send_json(200, thor_instance.observe(robot_ref, render_image=render_image))
        except ValueError as exc:
            self._send_json(400, {"status": "failed", "error": str(exc)})

    def _handle_reset(self):
        if not self._controller_ready():
            return
        try:
            payload = self._read_json()
        except Exception:
            payload = {}
        scene = payload.get("scene", thor_instance.scene)
        robot_count = payload.get("robots", payload.get("robot_count"))
        try:
            thor_instance.reset(scene=scene, robot_count=robot_count)
            self._send_json(200, {"status": "success", "scene": thor_instance.scene, "robots": len(thor_instance.robots), "state": thor_instance.capture_state()})
        except Exception as exc:
            traceback.print_exc()
            self._send_json(500, {"status": "failed", "error": str(exc)})

    @staticmethod
    def _query_value(query: dict, key: str):
        values = query.get(key)
        return values[0] if values else None

    @staticmethod
    def _query_bool(query: dict, key: str) -> bool:
        value = NativeControllerReceiverHandler._query_value(query, key)
        if value is None:
            return False
        return value.lower() in {"1", "true", "yes", "on"}

    def _send_json(self, status_code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status_code)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()


def main():
    global thor_instance
    parser = argparse.ArgumentParser(description="AI2-THOR Native Controller Multi-Agent Receiver HTTP Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", action="store_false", dest="headless")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--show", action="store_true", default=True)
    parser.add_argument("--no-show", action="store_false", dest="show")
    parser.add_argument("--display-fps", type=float, default=5.0)
    parser.add_argument("--robot0-dx", type=float, default=0.0)
    parser.add_argument("--robot0-dz", type=float, default=0.0)
    parser.add_argument("--robot0-left", type=float, default=0.0)
    parser.add_argument("--robot0-right", type=float, default=0.0)
    parser.add_argument("--robot0-back", type=float, default=0.0)
    parser.add_argument("--robot0-dyaw", type=float, default=0.0)
    parser.add_argument("--robot0-at-fridge", action="store_true")
    parser.add_argument("--robot0-fridge-distance", type=float, default=1.0)
    args = parser.parse_args()

    thor_instance = NativeControllerThorServer(
        scene=args.scene,
        robot_count=args.robots,
        headless=args.headless,
        width=args.width,
        height=args.height,
        robot0_dx=args.robot0_dx,
        robot0_dz=args.robot0_dz,
        robot0_left=args.robot0_left,
        robot0_right=args.robot0_right,
        robot0_back=args.robot0_back,
        robot0_dyaw=args.robot0_dyaw,
        robot0_at_fridge=args.robot0_at_fridge,
        robot0_fridge_distance=args.robot0_fridge_distance,
    )
    try:
        thor_instance.start()
    except Exception as exc:
        log_event("SERVER ERROR", f"Failed to start native Controller multi-agent AI2-THOR: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    server = ThreadingHTTPServer((args.host, args.port), NativeControllerReceiverHandler)
    log_event("SERVER OUT", f"http://{args.host}:{args.port}", blank_before=True)
    log_event("SERVER OUT", "mode: native_standard_controller_shared_unity")
    log_event("SERVER OUT", "GET  /robots")
    log_event("SERVER OUT", "GET  /state")
    log_event("SERVER OUT", "GET  /reachable_positions")
    log_event("SERVER OUT", "POST /observe")
    log_event("SERVER OUT", "POST /execute_actions")
    log_event("SERVER OUT", "POST /goto")
    log_event("SERVER OUT", "POST /reset")
    try:
        if args.show:
            server_thread = threading.Thread(target=server.serve_forever, daemon=True)
            server_thread.start()
            gui_started = thor_instance.display_loop(fps=args.display_fps)
            if not gui_started:
                log_event("SERVER OUT", "Running in --no-show fallback mode. Press Ctrl+C to stop.")
                while True:
                    time.sleep(1)
        else:
            server.serve_forever()
    except KeyboardInterrupt:
        log_event("SERVER IN", "KeyboardInterrupt: shutting down", blank_before=True)
    finally:
        server.shutdown()
        server.server_close()
        thor_instance.stop()


if __name__ == "__main__":
    main()
