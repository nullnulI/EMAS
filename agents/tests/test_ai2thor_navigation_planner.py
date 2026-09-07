from __future__ import annotations

import unittest
import threading
from types import MethodType

import ai2thor_receiver_server as module
from ai2thor_receiver_server import (
    NativeControllerReceiverHandler,
    NativeControllerThorServer,
    RobotState,
    astar_path,
    build_reachable_graph,
    choose_reachable_goal,
    failed_movement_edge,
    horizon_rotation_actions,
    held_object_safe_navigation_actions,
    parse_navigation_blocker,
    path_to_actions,
    remove_graph_edges,
    remove_graph_nodes,
    yaw_rotation_actions,
)


class NavigationPlannerPureFunctionTest(unittest.TestCase):
    def test_reachable_graph_connects_only_four_neighbors(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.25, "y": 0.9, "z": 0.0},
            {"x": 0.0, "y": 0.9, "z": 0.25},
            {"x": 0.25, "y": 0.9, "z": 0.25},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        self.assertIn(((0.25, 0.0), 0.25), graph[(0.0, 0.0)])
        self.assertIn(((0.0, 0.25), 0.25), graph[(0.0, 0.0)])
        self.assertNotIn((0.25, 0.25), [node for node, _ in graph[(0.0, 0.0)]])

    def test_astar_returns_shortest_grid_path(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.25, "y": 0.9, "z": 0.0},
            {"x": 0.5, "y": 0.9, "z": 0.0},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        self.assertEqual(astar_path(graph, (0.0, 0.0), (0.5, 0.0)), [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)])

    def test_astar_returns_none_when_disconnected(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 1.0, "y": 0.9, "z": 0.0},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        self.assertIsNone(astar_path(graph, (0.0, 0.0), (1.0, 0.0)))

    def test_remove_graph_nodes_blocks_dynamic_obstacle_but_preserves_start(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.25, "y": 0.9, "z": 0.0},
            {"x": 0.5, "y": 0.9, "z": 0.0},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        pruned = remove_graph_nodes(graph, {(0.25, 0.0), (0.0, 0.0)}, preserve_nodes={(0.0, 0.0)})

        self.assertIn((0.0, 0.0), pruned)
        self.assertNotIn((0.25, 0.0), pruned)
        self.assertNotIn((0.25, 0.0), [node for node, _ in pruned[(0.0, 0.0)]])

    def test_remove_graph_edges_blocks_failed_move_without_removing_nodes(self) -> None:
        positions = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.25, "y": 0.9, "z": 0.0},
            {"x": 0.0, "y": 0.9, "z": 0.25},
            {"x": 0.25, "y": 0.9, "z": 0.25},
        ]
        graph, _ = build_reachable_graph(positions, grid_size=0.25, epsilon=0.01)

        pruned = remove_graph_edges(graph, {((0.0, 0.0), (0.25, 0.0))})

        self.assertIn((0.0, 0.0), pruned)
        self.assertIn((0.25, 0.0), pruned)
        self.assertNotIn((0.25, 0.0), [node for node, _ in pruned[(0.0, 0.0)]])
        self.assertIn((0.0, 0.25), [node for node, _ in pruned[(0.0, 0.0)]])

    def test_failed_movement_edge_uses_pose_yaw_and_action(self) -> None:
        edge = failed_movement_edge(
            {
                "action": "MoveAhead",
                "robot_pose": {
                    "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                    "rotation": {"y": 90.0},
                },
            },
            grid_size=0.25,
        )

        self.assertEqual(edge, ((0.0, 0.0), (0.25, 0.0)))

    def test_parse_navigation_blocker_recognizes_agent_and_object(self) -> None:
        self.assertEqual(
            parse_navigation_blocker({"error": "Agent 1 is blocking Agent 0 from moving"}),
            {
                "failure_class": "agent_blocker",
                "blocking_robot_id": 1,
                "blocked_robot_id": 0,
                "error": "Agent 1 is blocking Agent 0 from moving",
            },
        )
        self.assertEqual(
            parse_navigation_blocker({"error": "Lettuce_abc is blocking Agent 1 from moving"})["failure_class"],
            "object_blocker",
        )
        fridge_blocker = parse_navigation_blocker(
            {
                "robot_id": 0,
                "error": "Fridge_e92350c6 is blocking the Agent from moving by (0.0000, 0.0000, -0.2500) with Lettuce_2d8f3ab9",
            }
        )
        self.assertEqual(fridge_blocker["failure_class"], "object_blocker")
        self.assertEqual(fridge_blocker["blocking_object_id"], "Fridge_e92350c6")
        self.assertEqual(fridge_blocker["blocked_robot_id"], 0)

    def test_path_to_actions_uses_yaw_and_moveahead(self) -> None:
        actions = path_to_actions(
            [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.25},
            ],
            current_yaw=0.0,
            rotate_step_degrees=90.0,
        )

        self.assertEqual(
            actions,
            [
                {"action": "RotateRight"},
                {"action": "MoveAhead"},
                {"action": "RotateLeft"},
                {"action": "MoveAhead"},
            ],
        )

    def test_yaw_rotation_actions_uses_precise_degrees_for_non_grid_yaw(self) -> None:
        actions = yaw_rotation_actions(current_yaw=90.0, target_yaw=41.1859)
        self.assertEqual(actions[0]["action"], "RotateLeft")
        self.assertAlmostEqual(actions[0]["degrees"], 48.8141)

    def test_horizon_rotation_actions_uses_precise_look_direction(self) -> None:
        self.assertEqual(
            horizon_rotation_actions(current_horizon=0.0, target_horizon=30.0),
            [{"action": "LookDown", "degrees": 30.0}],
        )
        self.assertEqual(
            horizon_rotation_actions(current_horizon=30.0, target_horizon=-10.0),
            [{"action": "LookUp", "degrees": 40.0}],
        )
        self.assertEqual(
            horizon_rotation_actions(current_horizon=30.0, target_horizon=30.0), []
        )

    def test_horizon_rotation_actions_quantizes_floating_point_noise(self) -> None:
        self.assertEqual(
            horizon_rotation_actions(
                current_horizon=30.00000762939453,
                target_horizon=60.0,
            ),
            [{"action": "LookDown", "degrees": 30.0}],
        )
        self.assertEqual(
            horizon_rotation_actions(
                current_horizon=29.99999237060547,
                target_horizon=0.0,
            ),
            [{"action": "LookUp", "degrees": 30.0}],
        )

    def test_path_to_actions_corrects_non_grid_initial_yaw(self) -> None:
        actions = path_to_actions(
            [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            current_yaw=41.1859,
            rotate_step_degrees=90.0,
        )

        self.assertEqual(actions[0]["action"], "RotateLeft")
        self.assertAlmostEqual(actions[0]["degrees"], 41.1859)
        self.assertEqual(actions[1], {"action": "MoveAhead"})

    def test_choose_reachable_goal_uses_standing_point_not_object_center(self) -> None:
        reachable = [
            {"x": 0.0, "y": 0.9, "z": 0.0},
            {"x": 0.75, "y": 0.9, "z": 0.0},
            {"x": 1.5, "y": 0.9, "z": 0.0},
        ]

        goal = choose_reachable_goal(
            reachable,
            {"x": 0.0, "y": 0.9, "z": 0.0},
            min_distance=0.5,
            max_distance=1.0,
        )

        self.assertEqual(goal, {"x": 0.75, "y": 0.9, "z": 0.0})

    def test_held_object_safe_navigation_splits_and_forces_rotations(self) -> None:
        actions = [
            {"action": "RotateLeft", "degrees": 48.0},
            {"action": "MoveAhead"},
        ]

        safe_actions = held_object_safe_navigation_actions(
            actions,
            holding_object=True,
            max_rotate_degrees=15.0,
        )

        self.assertEqual(
            safe_actions,
            [
                {"action": "RotateLeft", "degrees": 12.0, "forceAction": True},
                {"action": "RotateLeft", "degrees": 12.0, "forceAction": True},
                {"action": "RotateLeft", "degrees": 12.0, "forceAction": True},
                {"action": "RotateLeft", "degrees": 12.0, "forceAction": True},
                {"action": "MoveAhead"},
            ],
        )
        self.assertEqual(
            held_object_safe_navigation_actions(actions, holding_object=False),
            actions,
        )


class NavigationReceiverMethodTest(unittest.TestCase):
    def fake_server(self, *, yaw: float = 0.0) -> NativeControllerThorServer:
        server = NativeControllerThorServer.__new__(NativeControllerThorServer)
        server.lock = threading.RLock()
        server.robots = [
            RobotState(
                robot_id=0,
                name="Robot0",
                position={"x": 0.0, "y": 0.9, "z": 0.0},
                rotation={"x": 0.0, "y": yaw, "z": 0.0},
            )
        ]
        return server

    def test_controller_step_forces_render_image_when_action_recording_is_enabled(self) -> None:
        server = self.fake_server()

        class Controller:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return object()

        server.controller = Controller()
        server.render_action_frames = True

        server._controller_step({"action": "MoveAhead", "renderImage": False})

        self.assertTrue(server.controller.actions[0]["renderImage"])

    def test_controller_step_preserves_render_request_when_action_recording_is_disabled(self) -> None:
        server = self.fake_server()

        class Controller:
            def __init__(self):
                self.actions = []

            def step(self, action):
                self.actions.append(action)
                return object()

        server.controller = Controller()
        server.render_action_frames = False

        server._controller_step({"action": "MoveAhead", "renderImage": False})

        self.assertFalse(server.controller.actions[0]["renderImage"])

    def test_reachable_positions_response(self) -> None:
        server = self.fake_server()
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [{"x": 0.0, "y": 0.9, "z": 0.0}],
            server,
        )

        self.assertEqual(
            server.reachable_positions_response(0),
            {"status": "success", "robot_id": 0, "positions": [{"x": 0.0, "y": 0.9, "z": 0.0}]},
        )

    def test_goto_dry_run_returns_plan_without_executing(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
            ],
            server,
        )
        server.execute_batch = MethodType(lambda *args, **kwargs: self.fail("dry-run /goto must not execute"), server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "target_position": {"x": 0.25, "z": 0.0}})

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["execute"])
        self.assertEqual(result["actions"], [{"action": "RotateRight"}, {"action": "MoveAhead"}])
        self.assertNotIn("execute_result", result)

    def test_goto_dry_run_corrects_non_grid_initial_yaw(self) -> None:
        server = self.fake_server(yaw=41.1859)
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            server,
        )
        server.execute_batch = MethodType(lambda *args, **kwargs: self.fail("dry-run /goto must not execute"), server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "target_position": {"x": 0.0, "z": 0.25}})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actions"][0]["action"], "RotateLeft")
        self.assertAlmostEqual(result["actions"][0]["degrees"], 41.1859)
        self.assertEqual(result["actions"][1], {"action": "MoveAhead"})

    def test_goto_dry_run_faces_object_target_after_reaching_goal(self) -> None:
        server = self.fake_server(yaw=0.0)
        server.capture_state = MethodType(
            lambda self, robot_ref=None, render_image=False: {
                "objects": [
                    {
                        "id": "Target|1",
                        "type": "Target",
                        "position": {"x": 0.25, "y": 0.0, "z": 0.25},
                    }
                ]
            },
            server,
        )
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            server,
        )
        server.execute_batch = MethodType(lambda *args, **kwargs: self.fail("dry-run /goto must not execute"), server)

        result = server.goto({
            "task_id": "goto-1",
            "robot_id": 0,
            "object_type": "Target",
            "min_distance": 0.0,
            "require_target_visible": False,
        })

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actions"], [{"action": "MoveAhead"}, {"action": "RotateRight"}])
        self.assertTrue(result["face_target"])
        self.assertEqual(result["face_target_yaw"], 90.0)

    def test_goto_dry_run_can_disable_face_target(self) -> None:
        server = self.fake_server(yaw=0.0)
        server.capture_state = MethodType(
            lambda self, robot_ref=None, render_image=False: {
                "objects": [
                    {
                        "id": "Target|1",
                        "type": "Target",
                        "position": {"x": 0.25, "y": 0.0, "z": 0.25},
                    }
                ]
            },
            server,
        )
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            server,
        )
        server.execute_batch = MethodType(lambda *args, **kwargs: self.fail("dry-run /goto must not execute"), server)

        result = server.goto(
            {"task_id": "goto-1", "robot_id": 0, "object_type": "Target", "min_distance": 0.0, "face_target": False, "require_target_visible": False}
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["actions"], [{"action": "MoveAhead"}])
        self.assertFalse(result["face_target"])

    def test_plan_goto_uses_larger_default_distance_when_holding_object(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(
            lambda self, robot_ref=None, render_image=False: {
                "inventory": [{"objectId": "Lettuce|1", "objectType": "Lettuce"}],
                "objects": [
                    {
                        "id": "Fridge|1",
                        "type": "Fridge",
                        "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                    }
                ],
            },
            server,
        )
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.75},
                {"x": 0.0, "y": 0.9, "z": 1.0},
                {"x": 0.0, "y": 0.9, "z": 1.25},
            ],
            server,
        )
        server.robots[0].position = {"x": 0.0, "y": 0.9, "z": 1.25}

        result = server.plan_goto({
            "task_id": "goto-held-fridge",
            "robot_id": 0,
            "object_type": "Fridge",
            "require_target_visible": False,
        })

        self.assertEqual(result["goal_position"], {"x": 0.0, "y": 0.9, "z": 1.0})
        self.assertTrue(result["holding_object"])

    def test_plan_goto_avoids_learned_blocked_edge(self) -> None:
        server = self.fake_server(yaw=90.0)
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.5, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
                {"x": 0.25, "y": 0.9, "z": 0.25},
                {"x": 0.5, "y": 0.9, "z": 0.25},
            ],
            server,
        )

        result = server.plan_goto(
            {"task_id": "goto-1", "robot_id": 0, "target_position": {"x": 0.5, "z": 0.0}},
            blocked_edges={((0.0, 0.0), (0.25, 0.0))},
        )

        self.assertNotEqual(result["path"][1], {"x": 0.25, "y": 0.9, "z": 0.0})
        self.assertEqual(result["blocked_edge_count"], 1)

    def test_plan_goto_tries_alternate_target_near_goal_when_closest_is_unreachable(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.5, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
                {"x": 0.25, "y": 0.9, "z": 0.25},
                {"x": 0.5, "y": 0.9, "z": 0.25},
            ],
            server,
        )

        result = server.plan_goto(
            {
                "task_id": "goto-alternate-goal",
                "robot_id": 0,
                "target_position": {"x": 0.5, "z": 0.0},
                "min_distance": 0.2,
                "max_distance": 0.3,
            },
            blocked_edges={
                ((0.0, 0.0), (0.25, 0.0)),
                ((0.25, 0.25), (0.25, 0.0)),
                ((0.25, 0.0), (0.5, 0.0)),
            },
        )

        self.assertEqual(result["goal_position"], {"x": 0.5, "y": 0.9, "z": 0.25})
        self.assertEqual(result["blocked_edge_count"], 3)
        self.assertGreater(result["goal_candidate_count"], result["reachable_goal_candidate_count"])

    def test_plan_goto_avoids_other_robot_dynamic_obstacle(self) -> None:
        server = self.fake_server()
        server.robots.append(
            RobotState(
                robot_id=1,
                name="Robot1",
                position={"x": 0.25, "y": 0.9, "z": 0.0},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            )
        )
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.5, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
                {"x": 0.25, "y": 0.9, "z": 0.25},
                {"x": 0.5, "y": 0.9, "z": 0.25},
            ],
            server,
        )

        result = server.plan_goto(
            {
                "task_id": "goto-1",
                "robot_id": 0,
                "target_position": {"x": 0.5, "z": 0.0},
                "dynamic_obstacle_radius": 0.01,
            },
            dynamic_obstacles=server._dynamic_obstacles_for_robot(0),
        )

        self.assertNotIn({"x": 0.25, "y": 0.9, "z": 0.0}, result["path"])
        self.assertEqual(result["goal_position"], {"x": 0.5, "y": 0.9, "z": 0.0})
        self.assertEqual(result["dynamic_obstacles"][0]["robot_id"], 1)
        self.assertGreater(result["blocked_node_count"], 0)

    def test_goto_execute_calls_execute_batch_with_stop_on_failure_true(self) -> None:
        server = self.fake_server()
        calls = []
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
            ],
            server,
        )

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            calls.append(
                {
                    "actions": actions,
                    "default_robot_ref": default_robot_ref,
                    "render_image": render_image,
                    "stop_on_failure": stop_on_failure,
                }
            )
            return {"status": "success", "results": []}

        server.execute_batch = MethodType(execute_batch, server)

        result = server.goto(
            {
                "task_id": "goto-1",
                "robot_id": 0,
                "target_position": {"x": 0.0, "z": 0.25},
                "execute": True,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(calls[0]["actions"], [{"action": "MoveAhead"}])
        self.assertEqual(calls[0]["default_robot_ref"], 0)
        self.assertTrue(calls[0]["stop_on_failure"])

    def test_goto_execute_uses_safe_rotations_when_holding_object(self) -> None:
        server = self.fake_server()
        calls = []
        plan = {
            "status": "success",
            "robot_id": 0,
            "target": {"kind": "position"},
            "target_position": {"x": 0.25, "y": 0.9, "z": 0.0},
            "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
            "goal_position": {"x": 0.25, "y": 0.9, "z": 0.0},
            "path": [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
            ],
            "actions": [
                {"action": "RotateLeft", "degrees": 45.0},
                {"action": "MoveAhead"},
            ],
            "estimated_distance": 0.25,
            "face_target": False,
            "face_target_yaw": 0.0,
            "planner": "reachable_positions_astar",
            "grid_size": 0.25,
            "rotate_step_degrees": 90.0,
            "holding_object": True,
            "dynamic_obstacles": [],
            "blocked_node_count": 0,
        }
        server._plan_goto_with_dynamic_obstacles = MethodType(
            lambda self, payload, avoid_other_robots=True, **kwargs: dict(plan),
            server,
        )

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            calls.append(actions)
            return {
                "status": "success",
                "results": [
                    {"robot_id": default_robot_ref, "action": action["action"], "success": True}
                    for action in actions
                ],
            }

        server.execute_batch = MethodType(execute_batch, server)
        server._safe_capture_state = MethodType(lambda self, robot_ref=None: {}, server)

        result = server.goto(
            {
                "task_id": "goto-held",
                "robot_id": 0,
                "target_position": {"x": 0.25, "z": 0.0},
                "execute": True,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            calls[0],
            [
                {"action": "RotateLeft", "degrees": 15.0, "forceAction": True},
                {"action": "RotateLeft", "degrees": 15.0, "forceAction": True},
                {"action": "RotateLeft", "degrees": 15.0, "forceAction": True},
                {"action": "MoveAhead"},
            ],
        )
        self.assertTrue(result["replan_trace"][0]["holding_object_rotation_safety"])
        self.assertEqual(result["replan_trace"][0]["planned_action_count"], 2)
        self.assertEqual(result["replan_trace"][0]["action_count"], 4)

    def test_goto_replans_after_blocking_failure_and_succeeds(self) -> None:
        server = self.fake_server()
        server.robots.append(
            RobotState(
                robot_id=1,
                name="Robot1",
                position={"x": 0.25, "y": 0.9, "z": 0.0},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            )
        )
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        plan_calls = []
        plans = [
            {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "position"},
                "target_position": {"x": 0.5, "y": 0.9, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": 0.5, "y": 0.9, "z": 0.0},
                "path": [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.5, "y": 0.9, "z": 0.0}],
                "actions": [{"action": "MoveAhead"}],
                "estimated_distance": 0.5,
                "face_target": False,
                "face_target_yaw": 0.0,
                "planner": "reachable_positions_astar",
                "grid_size": 0.25,
                "rotate_step_degrees": 90.0,
                "dynamic_obstacles": [],
                "blocked_node_count": 0,
            },
            {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "position"},
                "target_position": {"x": 0.5, "y": 0.9, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": 0.5, "y": 0.9, "z": 0.0},
                "path": [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.0, "y": 0.9, "z": 0.25}, {"x": 0.5, "y": 0.9, "z": 0.25}],
                "actions": [{"action": "RotateRight"}, {"action": "MoveAhead"}],
                "estimated_distance": 0.75,
                "face_target": False,
                "face_target_yaw": 0.0,
                "planner": "reachable_positions_astar",
                "grid_size": 0.25,
                "rotate_step_degrees": 90.0,
                "dynamic_obstacles": [{"robot_id": 1}],
                "blocked_node_count": 1,
            },
        ]

        def plan_goto(self, payload, dynamic_obstacles=None, **kwargs):
            plan_calls.append(dynamic_obstacles)
            return plans.pop(0)

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            if len(plan_calls) == 1:
                return {
                    "status": "failed",
                    "results": [
                        {
                            "robot_id": 0,
                            "action": "MoveAhead",
                            "success": False,
                            "error": "Agent 1 is blocking Agent 0",
                        }
                    ],
                }
            return {"status": "success", "results": [{"robot_id": 0, "action": actions[0]["action"], "success": True}]}

        server.plan_goto = MethodType(plan_goto, server)
        server.execute_batch = MethodType(execute_batch, server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "target_position": {"x": 0.5, "z": 0.0}, "execute": True})

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(len(result["replan_trace"]), 2)
        self.assertEqual(plan_calls[0][0]["robot_id"], 1)
        self.assertEqual(result["execute_result"]["status"], "success")

    def test_goto_agent_blocker_attempts_yield_before_replan_success(self) -> None:
        server = self.fake_server(yaw=90.0)
        server.robots.append(
            RobotState(
                robot_id=1,
                name="Robot1",
                position={"x": 0.25, "y": 0.9, "z": 0.0},
                rotation={"x": 0.0, "y": 0.0, "z": 0.0},
            )
        )
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": [], "inventory": []}, server)
        server._safe_capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": [], "inventory": []}, server)
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [
                {"x": 0.0, "y": 0.9, "z": 0.0},
                {"x": 0.25, "y": 0.9, "z": 0.0},
                {"x": 0.5, "y": 0.9, "z": 0.0},
                {"x": 0.0, "y": 0.9, "z": 0.25},
                {"x": 0.25, "y": 0.9, "z": 0.25},
                {"x": 0.5, "y": 0.9, "z": 0.25},
                {"x": 0.25, "y": 0.9, "z": 0.5},
                {"x": 0.25, "y": 0.9, "z": 0.75},
            ],
            server,
        )
        calls = []

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            calls.append({"robot_id": default_robot_ref, "actions": actions})
            if len(calls) == 1:
                return {
                    "status": "failed",
                    "results": [
                        {
                            "robot_id": 0,
                            "action": "MoveAhead",
                            "success": False,
                            "error": "Agent 1 is blocking Agent 0 from moving",
                            "robot_pose": {
                                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                                "rotation": {"y": 90.0},
                            },
                        }
                    ],
                }
            if default_robot_ref == 1:
                server.robots[1].position = {"x": 0.25, "y": 0.9, "z": 0.75}
            return {
                "status": "success",
                "results": [
                    {"robot_id": default_robot_ref, "action": action["action"], "success": True}
                    for action in actions
                ],
            }

        server.execute_batch = MethodType(execute_batch, server)

        result = server.goto(
            {
                "task_id": "goto-yield",
                "robot_id": 0,
                "target_position": {"x": 0.5, "z": 0.0},
                "execute": True,
                "avoid_other_robots": False,
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["negotiation_trace"][0]["status"], "success")
        self.assertEqual(result["negotiation_trace"][0]["blocking_robot_id"], 1)
        self.assertTrue(any(call["robot_id"] == 1 for call in calls))
        self.assertGreaterEqual(len(result["learned_blocked_edges"]), 1)

    def test_goto_fails_after_replan_limit(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)

        def plan_goto(self, payload, dynamic_obstacles=None, **kwargs):
            return {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "position"},
                "target_position": {"x": 0.25, "y": 0.9, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": 0.25, "y": 0.9, "z": 0.0},
                "path": [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.25, "y": 0.9, "z": 0.0}],
                "actions": [{"action": "MoveAhead"}],
                "estimated_distance": 0.25,
                "face_target": False,
                "face_target_yaw": 0.0,
                "planner": "reachable_positions_astar",
                "grid_size": 0.25,
                "rotate_step_degrees": 90.0,
                "dynamic_obstacles": [],
                "blocked_node_count": 0,
            }

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            return {
                "status": "failed",
                "results": [
                    {
                        "robot_id": 0,
                        "action": "MoveAhead",
                        "success": False,
                        "error": "Agent 1 is blocking Agent 0",
                    }
                ],
            }

        server.plan_goto = MethodType(plan_goto, server)
        server.execute_batch = MethodType(execute_batch, server)

        result = server.goto(
            {
                "task_id": "goto-1",
                "robot_id": 0,
                "target_position": {"x": 0.25, "z": 0.0},
                "execute": True,
                "max_replans": 1,
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "execution_failed_after_replans")
        self.assertEqual(result["replan_count"], 1)
        self.assertEqual(len(result["replan_trace"]), 2)
        self.assertIn("blocking", result["failed_action"]["error"])

    def test_goto_replan_failure_survives_broken_capture_state(self) -> None:
        server = self.fake_server()
        plan_calls = 0

        def plan_goto(self, payload, dynamic_obstacles=None, **kwargs):
            nonlocal plan_calls
            plan_calls += 1
            if plan_calls > 1:
                raise module.NavigationPlanningError(
                    "no_reachable_positions",
                    "GetReachablePositions returned no usable positions",
                )
            return {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "position"},
                "target_position": {"x": 0.25, "y": 0.9, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": 0.25, "y": 0.9, "z": 0.0},
                "path": [{"x": 0.0, "y": 0.9, "z": 0.0}, {"x": 0.25, "y": 0.9, "z": 0.0}],
                "actions": [{"action": "MoveAhead"}],
                "estimated_distance": 0.25,
                "face_target": False,
                "face_target_yaw": 0.0,
                "planner": "reachable_positions_astar",
                "grid_size": 0.25,
                "rotate_step_degrees": 90.0,
                "dynamic_obstacles": [],
                "blocked_node_count": 0,
            }

        def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
            return {
                "status": "failed",
                "results": [
                    {
                        "robot_id": 0,
                        "action": "MoveAhead",
                        "success": False,
                        "error": "Agent 1 is blocking Agent 0",
                    }
                ],
            }

        def capture_state(self, robot_ref=None, render_image=False):
            raise ValueError("write to closed file")

        server.plan_goto = MethodType(plan_goto, server)
        server.execute_batch = MethodType(execute_batch, server)
        server.capture_state = MethodType(capture_state, server)

        result = server.goto(
            {
                "task_id": "goto-1",
                "robot_id": 0,
                "target_position": {"x": 0.25, "z": 0.0},
                "execute": True,
            }
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "execution_failed_after_replans")
        self.assertEqual(result["replan_error_code"], "no_reachable_positions")
        self.assertEqual(result["execute_result"]["state"]["status"], "unavailable")
        self.assertEqual(result["execute_result"]["state"]["error_type"], "ValueError")

    def test_execute_batch_uses_safe_capture_state(self) -> None:
        server = self.fake_server()

        def execute(self, robot_ref, action, render_image=False, **kwargs):
            return {
                "robot_id": robot_ref,
                "action": action,
                "success": True,
                "error": None,
            }

        def capture_state(self, robot_ref=None, render_image=False):
            raise RuntimeError("controller pipe closed")

        server.execute = MethodType(execute, server)
        server.capture_state = MethodType(capture_state, server)

        result = server.execute_batch([{"action": "MoveAhead"}], default_robot_ref=0)

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["state"]["status"], "unavailable")
        self.assertEqual(result["state"]["error_type"], "RuntimeError")

    def test_execute_batch_passes_through_all_documented_native_actions(self) -> None:
        server = self.fake_server()
        captured = []

        def execute(self, robot_ref, action, render_image=False, **kwargs):
            captured.append((robot_ref, action, kwargs))
            return {
                "robot_id": robot_ref,
                "action": action,
                "success": True,
                "error": None,
            }

        server.execute = MethodType(execute, server)
        server.capture_state = MethodType(
            lambda self, robot_ref=None, render_image=False: {}, server
        )
        actions = [
            {"action": "MoveAhead", "moveMagnitude": 0.25},
            {"action": "MoveBack", "moveMagnitude": 0.25},
            {"action": "MoveLeft", "moveMagnitude": 0.25},
            {"action": "MoveRight", "moveMagnitude": 0.25},
            {"action": "RotateLeft", "degrees": 90},
            {"action": "RotateRight", "degrees": 90},
            {"action": "LookUp", "degrees": 30},
            {"action": "LookDown", "degrees": 30},
            {
                "action": "Teleport",
                "position": {"x": 0, "y": 0.9, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "forceAction": True,
            },
            {
                "action": "TeleportFull",
                "position": {"x": 0, "y": 0.9, "z": 0},
                "rotation": {"x": 0, "y": 0, "z": 0},
                "horizon": 0,
                "standing": True,
                "forceAction": True,
            },
            {"action": "GetReachablePositions"},
            {"action": "PickupObject", "objectId": "Mug|1", "forceAction": True},
            {"action": "PutObject", "objectId": "Sink|1", "forceAction": True},
            {"action": "OpenObject", "objectId": "Cabinet|1", "forceAction": True},
            {"action": "CloseObject", "objectId": "Cabinet|1", "forceAction": True},
            {"action": "DropHandObject"},
            {"action": "PushObject", "objectId": "Chair|1", "moveMagnitude": 200.0},
            {"action": "PullObject", "objectId": "Chair|1", "moveMagnitude": 200.0},
            {"action": "MoveHeldObject", "right": 0.1, "up": 0.0, "ahead": 0.0},
            {"action": "SliceObject", "objectId": "Tomato|1"},
            {"action": "BreakObject", "objectId": "Plate|1"},
            {"action": "CookObject", "objectId": "Potato|1"},
            {"action": "CleanObject", "objectId": "Plate|1"},
            {
                "action": "FillObjectWithLiquid",
                "objectId": "Mug|1",
                "fillLiquid": "water",
            },
            {
                "action": "SetObjectStates",
                "objectType": "Mug",
                "stateChanges": [{"stateChange": "isDirty", "value": False}],
            },
            {"action": "Pass"},
            {"action": "Done"},
        ]

        result = server.execute_batch(actions, default_robot_ref=0)

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(captured), 27)
        self.assertEqual([item[1] for item in captured], [item["action"] for item in actions])
        self.assertEqual(captured[18][2], {"right": 0.1, "up": 0.0, "ahead": 0.0})
        self.assertEqual(captured[23][2]["fillLiquid"], "water")
        self.assertEqual(
            captured[24][2]["stateChanges"],
            [{"stateChange": "isDirty", "value": False}],
        )

    def test_drop_and_move_held_report_the_held_object_as_interacted(self) -> None:
        server = NativeControllerThorServer.__new__(NativeControllerThorServer)
        before = {
            "objects": [],
            "inventoryObjects": [{"objectId": "Mug|1", "objectType": "Mug"}],
        }
        after_move = {
            "objects": [],
            "inventoryObjects": [{"objectId": "Mug|1", "objectType": "Mug"}],
        }
        after_drop = {
            "objects": [{"objectId": "Mug|1", "objectType": "Mug"}],
            "inventoryObjects": [],
        }

        moved = server._interacted_objects_after_action(
            "MoveHeldObject", {}, before, after_move
        )
        dropped = server._interacted_objects_after_action(
            "DropHandObject", {}, before, after_drop
        )

        self.assertEqual([item["objectId"] for item in moved], ["Mug|1"])
        self.assertEqual([item["objectId"] for item in dropped], ["Mug|1"])

    def test_controller_step_is_serialized(self) -> None:
        server = self.fake_server()
        entered_first = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()
        order: list[str] = []
        active = 0
        max_active = 0
        guard = threading.Lock()

        class FakeController:
            def step(self, action):
                nonlocal active, max_active
                with guard:
                    active += 1
                    max_active = max(max_active, active)
                    order.append(action["action"])
                if action["action"] == "first":
                    entered_first.set()
                    release_first.wait(timeout=2.0)
                else:
                    second_entered.set()
                with guard:
                    active -= 1
                return {"action": action["action"]}

        server.controller = FakeController()

        first_result = {}
        second_result = {}

        def run_first() -> None:
            first_result["value"] = server._controller_step({"action": "first"})

        def run_second() -> None:
            second_result["value"] = server._controller_step({"action": "second"})

        t1 = threading.Thread(target=run_first)
        t2 = threading.Thread(target=run_second)
        t1.start()
        self.assertTrue(entered_first.wait(timeout=2.0))
        t2.start()
        self.assertFalse(second_entered.wait(timeout=0.2))
        self.assertEqual(order, ["first"])
        self.assertEqual(max_active, 1)
        release_first.set()
        t1.join(timeout=2.0)
        t2.join(timeout=2.0)

        self.assertFalse(t1.is_alive())
        self.assertFalse(t2.is_alive())
        self.assertEqual(order, ["first", "second"])
        self.assertEqual(max_active, 1)
        self.assertEqual(first_result["value"], {"action": "first"})
        self.assertEqual(second_result["value"], {"action": "second"})

    def test_goto_object_type_not_found(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(lambda self, robot_ref=None, render_image=False: {"objects": []}, server)
        server._get_reachable_positions = MethodType(lambda self, robot_ref=None: [], server)

        result = server.goto({"task_id": "goto-1", "robot_id": 0, "object_type": "Fridge"})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "target_not_found")

    def test_handler_health_reports_service_without_controller_ready_check(self) -> None:
        class FakeThor:
            robots = [object(), object()]

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler.path = "/health"
            sent = []
            handler._send_json = MethodType(lambda self, code, payload: sent.append((code, payload)), handler)

            handler.do_GET()
        finally:
            module.thor_instance = original

        self.assertEqual(sent, [(200, {"status": "ok", "service": "ai2thor_receiver_server", "robots": 2})])

    def test_handler_goto_dispatches_to_thor_instance(self) -> None:
        class FakeThor:
            controller = object()

            def goto(self, payload):
                self.payload = payload
                return {"status": "success", "actions": [{"action": "MoveAhead"}]}

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler._controller_ready = MethodType(lambda self: True, handler)
            handler._read_json = MethodType(lambda self: {"task_id": "goto-1", "execute": False}, handler)
            sent = []
            handler._send_json = MethodType(lambda self, code, payload: sent.append((code, payload)), handler)

            handler._handle_goto()
        finally:
            module.thor_instance = original

        self.assertEqual(sent, [(200, {"status": "success", "actions": [{"action": "MoveAhead"}]})])
        self.assertEqual(fake.payload, {"task_id": "goto-1", "execute": False})

    def test_handler_execute_dispatches_to_thor_instance(self) -> None:
        class FakeThor:
            controller = object()

            def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
                self.call = {
                    "actions": actions,
                    "default_robot_ref": default_robot_ref,
                    "render_image": render_image,
                    "stop_on_failure": stop_on_failure,
                }
                return {"status": "success", "results": [{"index": 0, "robot_id": 0, "action": "Pass", "success": True}]}

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler._controller_ready = MethodType(lambda self: True, handler)
            handler._read_json = MethodType(
                lambda self: {
                    "task_id": "exec-1",
                    "robot_id": 0,
                    "stop_on_failure": False,
                    "actions": [{"action": "Pass"}],
                },
                handler,
            )
            sent = []
            handler._send_preencoded_json = MethodType(lambda self, code, body: sent.append((code, module.json.loads(body.decode("utf-8")))), handler)

            handler._handle_execute()
        finally:
            module.thor_instance = original

        self.assertEqual(fake.call["default_robot_ref"], 0)
        self.assertFalse(fake.call["stop_on_failure"])
        self.assertEqual(sent[0][0], 200)
        self.assertEqual(sent[0][1]["status"], "success")
        self.assertEqual(sent[0][1]["task_id"], "exec-1")

    def test_handler_execute_returns_json_for_receiver_exception(self) -> None:
        class FakeThor:
            controller = object()

            def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
                raise RuntimeError("controller pipe closed")

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler._controller_ready = MethodType(lambda self: True, handler)
            handler._read_json = MethodType(lambda self: {"task_id": "exec-err", "actions": [{"action": "OpenObject"}]}, handler)
            sent = []
            handler._send_json = MethodType(lambda self, code, payload: sent.append((code, payload)), handler)

            handler._handle_execute()
        finally:
            module.thor_instance = original

        self.assertEqual(sent[0][0], 500)
        self.assertEqual(sent[0][1]["status"], "failed")
        self.assertEqual(sent[0][1]["task_id"], "exec-err")
        self.assertEqual(sent[0][1]["error_code"], "receiver_exception")
        self.assertEqual(sent[0][1]["error_type"], "RuntimeError")
        self.assertIn("controller pipe closed", sent[0][1]["error"])

    def test_handler_execute_returns_serialization_error_without_compact_fallback(self) -> None:
        class FakeThor:
            controller = object()

            def execute_batch(self, actions, default_robot_ref=None, render_image=False, stop_on_failure=True):
                return {
                    "status": "success",
                    "results": [
                        {
                            "index": 0,
                            "robot_id": 0,
                            "action": "PickupObject",
                            "success": True,
                            "error": None,
                            "non_serializable": object(),
                        }
                    ],
                }

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler._controller_ready = MethodType(lambda self: True, handler)
            handler._read_json = MethodType(lambda self: {"task_id": "exec-strict", "actions": [{"action": "PickupObject"}]}, handler)
            sent = []
            handler._send_preencoded_json = MethodType(lambda self, code, body: sent.append((code, body)), handler)

            handler._handle_execute()
        finally:
            module.thor_instance = original

        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0][0], 500)
        payload = module.json.loads(sent[0][1].decode("utf-8"))
        self.assertEqual(payload["status"], "failed")
        self.assertEqual(payload["task_id"], "exec-strict")
        self.assertEqual(payload["error_code"], "execute_result_not_serializable")
        self.assertEqual(payload["error_type"], "TypeError")
        self.assertIn("not JSON serializable", payload["error"])
        self.assertNotIn("response_compacted", payload)

    def test_strict_jsonable_rejects_bad_item_without_recursing(self) -> None:
        class BadItem:
            def item(self):
                return BadItem()

        with self.assertRaises(TypeError):
            NativeControllerReceiverHandler._strict_json_body({"value": BadItem()})

    def test_interacted_objects_after_action_does_not_self_reference_after_state(self) -> None:
        server = NativeControllerThorServer.__new__(NativeControllerThorServer)
        object_id = "Fridge|1"
        before_meta = {
            "objects": [
                {
                    "objectId": object_id,
                    "objectType": "Fridge",
                    "openable": True,
                    "isOpen": True,
                }
            ]
        }
        after_meta = {
            "objects": [
                {
                    "objectId": object_id,
                    "objectType": "Fridge",
                    "openable": True,
                    "isOpen": False,
                }
            ]
        }

        interacted = server._interacted_objects_after_action(
            "CloseObject",
            {"objectId": object_id},
            before_meta,
            after_meta,
        )

        self.assertEqual(len(interacted), 1)
        self.assertIsNot(interacted[0], interacted[0]["after"])
        self.assertFalse(interacted[0]["after"]["isOpen"])
        NativeControllerReceiverHandler._strict_json_body({"results": [{"interacted_objects": interacted}]})

    def test_strict_execute_result_json_preserves_state_change_fields(self) -> None:
        raw = {
            "status": "success",
            "task_id": "exec-state",
            "results": [
                {
                    "index": 0,
                    "robot_id": 0,
                    "action": "PickupObject",
                    "success": True,
                    "error": None,
                    "inventory": [{"objectId": "Tomato|1", "objectType": "Tomato"}],
                    "held_object": {"objectId": "Tomato|1", "objectType": "Tomato"},
                    "robot_pose_changed": True,
                    "robot_pose_delta": {"before": {"horizon": 0.0}, "after": {"horizon": 30.0}},
                    "interacted_objects": [
                        {
                            "objectId": "Tomato|1",
                            "after": {"objectId": "Tomato|1", "objectType": "Tomato", "inInventory": True},
                            "state_changed": True,
                        }
                    ],
                }
            ],
        }

        body = NativeControllerReceiverHandler._strict_json_body(raw)
        payload = module.json.loads(body.decode("utf-8"))

        result = payload["results"][0]
        self.assertEqual(result["inventory"], [{"objectId": "Tomato|1", "objectType": "Tomato"}])
        self.assertEqual(result["held_object"], {"objectId": "Tomato|1", "objectType": "Tomato"})
        self.assertEqual(result["robot_pose_delta"], {"before": {"horizon": 0.0}, "after": {"horizon": 30.0}})
        self.assertEqual(result["interacted_objects"][0]["after"]["inInventory"], True)
        self.assertNotIn("response_compacted", payload)

    def test_handler_goto_returns_json_for_receiver_exception(self) -> None:
        class FakeThor:
            controller = object()

            def goto(self, payload):
                raise RuntimeError("controller pipe closed")

        fake = FakeThor()
        original = module.thor_instance
        module.thor_instance = fake
        try:
            handler = NativeControllerReceiverHandler.__new__(NativeControllerReceiverHandler)
            handler._controller_ready = MethodType(lambda self: True, handler)
            handler._read_json = MethodType(lambda self: {"task_id": "goto-1", "execute": True}, handler)
            sent = []
            handler._send_json = MethodType(lambda self, code, payload: sent.append((code, payload)), handler)

            handler._handle_goto()
        finally:
            module.thor_instance = original

        self.assertEqual(sent[0][0], 500)
        self.assertEqual(sent[0][1]["status"], "failed")
        self.assertEqual(sent[0][1]["task_id"], "goto-1")
        self.assertEqual(sent[0][1]["error_code"], "receiver_exception")
        self.assertEqual(sent[0][1]["error_type"], "RuntimeError")
        self.assertIn("controller pipe closed", sent[0][1]["error"])


    def test_plan_goto_uses_interactable_pose_horizon_for_floor_target(self) -> None:
        server = self.fake_server(yaw=0.0)
        server.capture_state = MethodType(
            lambda self, robot_ref=None, render_image=False: {
                "objects": [{
                    "id": "ButterKnife|1",
                    "type": "ButterKnife",
                    "position": {"x": 0.25, "y": 0.05, "z": 0.0},
                    "visible": False,
                }]
            },
            server,
        )
        reachable = [{"x": 0.0, "y": 0.9, "z": 0.0}]
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: reachable,
            server,
        )
        server._get_interactable_poses = MethodType(
            lambda self, robot_ref, object_id, positions, max_distance=None: [{
                "x": 0.0,
                "y": 0.9,
                "z": 0.0,
                "rotation": 0.0,
                "horizon": 30.0,
                "standing": True,
            }],
            server,
        )

        result = server.plan_goto({
            "robot_id": 0,
            "object_type": "ButterKnife",
            "min_distance": 0.0,
        })

        self.assertEqual(result["actions"], [{"action": "LookDown", "degrees": 30.0}])
        self.assertTrue(result["require_target_visible"])
        self.assertEqual(result["target_pose_source"], "ai2thor_get_interactable_poses")
        self.assertEqual(result["interactable_pose_count"], 1)
        self.assertEqual(result["goal_horizon"], 30.0)
        self.assertEqual(result["goal_rotation"], 0.0)

    def test_plan_goto_returns_no_interactable_pose(self) -> None:
        server = self.fake_server()
        server.capture_state = MethodType(
            lambda self, robot_ref=None, render_image=False: {
                "objects": [{
                    "id": "Bowl|1",
                    "type": "Bowl",
                    "position": {"x": 0.25, "y": 0.05, "z": 0.0},
                }]
            },
            server,
        )
        server._get_reachable_positions = MethodType(
            lambda self, robot_ref=None: [{"x": 0.0, "y": 0.9, "z": 0.0}],
            server,
        )
        server._get_interactable_poses = MethodType(
            lambda self, robot_ref, object_id, positions, max_distance=None: [],
            server,
        )

        result = server.goto({"robot_id": 0, "object_type": "Bowl"})

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "no_interactable_pose")

    def test_goto_retries_pose_and_fails_when_target_stays_invisible(self) -> None:
        server = self.fake_server()
        plan_calls = []

        def plan(self, payload, avoid_other_robots=True, excluded_target_poses=None, **kwargs):
            excluded = set(excluded_target_poses or set())
            plan_calls.append(excluded)
            pose_index = len(excluded)
            pose = {
                "x": float(pose_index),
                "y": 0.9,
                "z": 0.0,
                "rotation": 0.0,
                "horizon": 30.0,
            }
            return {
                "status": "success",
                "robot_id": 0,
                "target": {"kind": "object", "object_id": "Mug|1"},
                "target_position": {"x": 0.0, "y": 0.05, "z": 0.0},
                "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "goal_position": {"x": pose["x"], "y": 0.9, "z": 0.0},
                "goal_pose": pose,
                "path": [],
                "actions": [],
                "rotate_step_degrees": 90.0,
                "holding_object": False,
                "require_target_visible": True,
                "dynamic_obstacles": [],
                "blocked_node_count": 0,
                "blocked_edge_count": 0,
            }

        server._plan_goto_with_dynamic_obstacles = MethodType(plan, server)
        server._safe_capture_state = MethodType(
            lambda self, robot_ref=None: {
                "objects": [{"id": "Mug|1", "visible": False}]
            },
            server,
        )

        result = server.goto({
            "robot_id": 0,
            "object_type": "Mug",
            "execute": True,
            "max_replans": 1,
        })

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_code"], "target_not_visible_after_goto")
        self.assertFalse(result["post_target_visible"])
        self.assertEqual(result["post_target_object_id"], "Mug|1")
        self.assertEqual(len(plan_calls), 2)
        self.assertEqual(len(plan_calls[1]), 1)

    def test_goto_succeeds_only_after_exact_target_is_visible(self) -> None:
        server = self.fake_server()
        plan = {
            "status": "success",
            "robot_id": 0,
            "target": {"kind": "object", "object_id": "Mug|1"},
            "target_position": {"x": 0.0, "y": 0.05, "z": 0.0},
            "start_position": {"x": 0.0, "y": 0.9, "z": 0.0},
            "goal_position": {"x": 0.0, "y": 0.9, "z": 0.0},
            "goal_pose": {
                "x": 0.0, "y": 0.9, "z": 0.0, "rotation": 0.0, "horizon": 30.0
            },
            "path": [],
            "actions": [],
            "rotate_step_degrees": 90.0,
            "holding_object": False,
            "require_target_visible": True,
            "dynamic_obstacles": [],
            "blocked_node_count": 0,
            "blocked_edge_count": 0,
        }
        server._plan_goto_with_dynamic_obstacles = MethodType(
            lambda self, payload, **kwargs: dict(plan),
            server,
        )
        server._safe_capture_state = MethodType(
            lambda self, robot_ref=None: {
                "objects": [
                    {"id": "Mug|other", "visible": False},
                    {"id": "Mug|1", "visible": True},
                ]
            },
            server,
        )

        result = server.goto({
            "robot_id": 0,
            "object_type": "Mug",
            "execute": True,
        })

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["post_target_visible"])
        self.assertEqual(result["post_target_object_id"], "Mug|1")

if __name__ == "__main__":
    unittest.main()
