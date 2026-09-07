from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.tests.test_hybrid_task_service import load_module


def _task(task_id: str, depends_on: tuple[str, ...] = ()) -> dict:
    return {
        "id": task_id,
        "name": f"Task {task_id}",
        "description": f"Execute task {task_id}",
        "action": "navigate",
        "grounding": {
            "source_object_tags": ["Target"],
            "source_object_ids": [f"Target|{task_id}"],
        },
        "depends_on": list(depends_on),
    }


def _graph(*tasks: dict) -> dict:
    return {
        "task": "integration task",
        "flat_tasks": [deepcopy(task) for task in tasks],
        "dependency_edges": [
            {"from": dependency, "to": task["id"]}
            for task in tasks
            for dependency in task.get("depends_on") or []
        ],
        "root_task_ids": [
            task["id"] for task in tasks if not task.get("depends_on")
        ],
        "chains": [],
    }


def _agent(agent_id: str, *, observation_epoch: int = 0) -> dict:
    return {
        "agent_id": agent_id,
        "skills": ["navigate"],
        "held_objects": [],
        "visible_objects": [],
        "observation_epoch": observation_epoch,
    }


def _args(output_dir: Path, planning_chat: object) -> argparse.Namespace:
    return argparse.Namespace(
        run_name="execution_plan_integration",
        scene_name="FloorPlan1",
        output_dir=output_dir,
        task="integration task",
        agent_skills_json=None,
        agentnum=2,
        disable_qwen=False,
        _shared_planning_chat=planning_chat,
        planning_model_path="unused-model",
        qwen_conv_mode="unused-mode",
        qwen_num_gpus=1,
        planning_max_new_tokens=128,
        planning_max_attempts=3,
        execution_mode="task_service",
        adapter_result_file=None,
        adapter_auto_success=False,
        max_task_loops=3,
        max_task_retries=3,
        save_intermediates=True,
        enable_task_graph_replan=True,
        max_task_graph_replans=3,
        max_task_replans_per_source=1,
        task_graph_replan_max_attempts=3,
    )


def _first_execution_unit(execution_plan: dict, task_graph: dict) -> list[dict]:
    if execution_plan.get("state") != "dispatchable":
        return []
    tasks = {str(task["id"]): task for task in task_graph["flat_tasks"]}
    return [
        {
            "subtask": deepcopy(tasks[str(assignment["task_id"])]),
            "agent_id": str(assignment["agent_id"]),
        }
        for assignment in execution_plan["units"][0]["assignments"]
    ]


class RecordingAdapter:
    instances: list["RecordingAdapter"] = []

    def __init__(self, unused_args: argparse.Namespace) -> None:
        self.sent: list[dict] = []
        self.received: list[dict] = []
        type(self).instances.append(self)

    def send_task_allocation(self, payload: dict, unused_loop_dir: Path) -> None:
        self.sent.append(deepcopy(payload))

    def receive_execution_report(self, payload: dict, unused_loop_dir: Path) -> dict:
        self.received.append(deepcopy(payload))
        epoch = len(self.received)
        statuses = [
            {
                "subtask_id": str(item["subtask"]["id"]),
                "agent_id": str(item["agent_id"]),
                "status": "success",
                "subtask": deepcopy(item["subtask"]),
            }
            for item in payload["assignments"]
        ]
        return {
            "task_statuses": statuses,
            "execution": {
                "completed_task_ids": [item["subtask_id"] for item in statuses],
                "traces": [],
            },
            "agent_states": [
                _agent("0", observation_epoch=epoch),
                _agent("1", observation_epoch=epoch),
            ],
            "object_changes": [],
            "feedback": [],
        }


class NeverDispatchAdapter(RecordingAdapter):
    def send_task_allocation(self, payload: dict, unused_loop_dir: Path) -> None:
        raise AssertionError(f"blocked Execution Plan was dispatched: {payload}")

    def receive_execution_report(self, payload: dict, unused_loop_dir: Path) -> dict:
        raise AssertionError(f"blocked Execution Plan requested a report: {payload}")


def _patch_loop_boundaries(
    module,
    *,
    graph: dict,
    build_execution_plan,
    adapter_type,
    replan_from_execution_report,
) -> ExitStack:
    stack = ExitStack()
    replacements = {
        "initialize_scenegraph_and_agents": lambda *unused: (
            None,
            {},
            [_agent("0"), _agent("1")],
            {"source": "mock"},
        ),
        "build_initial_task_graph": lambda *unused, **unused_kwargs: {
            "subgraph": {},
            "subgraph_path": None,
            "task_graph": deepcopy(graph),
            "task_graph_path": "mock_initial_task_graph.json",
        },
        "apply_scene_catalog_selectors": lambda task_graph, unused_catalog: task_graph,
        "rebuild_task_graph": lambda task_graph: task_graph,
        "initial_completed_task_ids": lambda unused_graph: set(),
        "extract_subgraph_for_task": lambda *unused: {"nodes": [], "triples": []},
        "merge_subgraphs_for_allocation": lambda *unused, **unused_kwargs: {},
        "build_execution_plan": build_execution_plan,
        "first_execution_unit": _first_execution_unit,
        "TaskExecutionServiceAdapter": adapter_type,
        "replan_from_execution_report": replan_from_execution_report,
        "update_scenegraph_from_report": lambda current, unused_report, unused_dir: current,
        "save_json": lambda *unused, **unused_kwargs: None,
        "save_intermediate_json": lambda *unused, **unused_kwargs: None,
        "save_allocation_failure": lambda *unused, **unused_kwargs: None,
    }
    for name, replacement in replacements.items():
        stack.enter_context(patch.object(module, name, replacement))
    return stack


class HybridExecutionPlanLoopTest(unittest.TestCase):
    def setUp(self) -> None:
        RecordingAdapter.instances.clear()
        NeverDispatchAdapter.instances.clear()
        self.module = load_module()

    def test_three_invalid_allocation_attempts_trigger_planning_without_dispatch(self) -> None:
        graph = _graph(_task("T1"))
        planning_chat = object()
        plan_calls: list[dict] = []
        replan_calls: list[dict] = []

        def blocked_plan(**kwargs) -> dict:
            plan_calls.append(deepcopy(kwargs))
            attempts = [
                {
                    "attempt": attempt,
                    "validation": {
                        "status": "invalid",
                        "errors": [f"invalid complete plan {attempt}"],
                    },
                }
                for attempt in range(1, 4)
            ]
            return {
                "state": "blocked",
                "task_graph_version": kwargs["task_graph_version"],
                "graph_fingerprint": "sha256:invalid-three-times",
                "execution_policy": "first_unit_then_replan",
                "units": [],
                "dependency_edges": [],
                "blocking": {
                    "code": "invalid_model_plan",
                    "task_ids": ["T1"],
                    "agent_inventories": {"0": [], "1": []},
                    "conflicts": ["invalid complete plan 3"],
                    "recommended_recovery": "replan_task_graph",
                },
                "diagnostics": {
                    "attempts": attempts,
                    "remaining_task_ids": ["T1"],
                    "fallback_used": False,
                },
            }

        def failed_replan(**kwargs) -> dict:
            replan_calls.append(deepcopy(kwargs))
            return {
                "status": "failed",
                "reason": "mock planning rejected replacement graph",
            }

        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory), planning_chat)
            with _patch_loop_boundaries(
                self.module,
                graph=graph,
                build_execution_plan=blocked_plan,
                adapter_type=NeverDispatchAdapter,
                replan_from_execution_report=failed_replan,
            ):
                with self.assertRaises(self.module.TaskAllocationError) as captured:
                    self.module.run_hybrid_loop(args)

        self.assertEqual(captured.exception.code, "allocation_replan_failed")
        self.assertEqual(len(plan_calls), 1)
        self.assertEqual(plan_calls[0]["allocation_max_attempts"], 3)
        self.assertEqual(plan_calls[0]["progress"], {
            "completed_task_ids": [],
            "failed_task_ids": [],
        })
        self.assertEqual(len(replan_calls), 1)
        report = replan_calls[0]["report"]
        self.assertEqual(report["trigger"], "allocation_blocked")
        self.assertEqual(
            len(report["execution_plan"]["diagnostics"]["attempts"]),
            3,
        )
        self.assertEqual(report["task_statuses"][0]["failure_code"], "invalid_model_plan")
        self.assertEqual(len(NeverDispatchAdapter.instances), 1)
        self.assertEqual(NeverDispatchAdapter.instances[0].sent, [])
        self.assertEqual(NeverDispatchAdapter.instances[0].received, [])

    def test_blocked_plan_respects_episode_and_source_replan_budgets(self) -> None:
        cases = [
            {
                "name": "episode_budget",
                "max_task_graph_replans": 0,
                "max_task_replans_per_source": 1,
                "reason": "episode_replan_budget_exhausted",
            },
            {
                "name": "source_budget",
                "max_task_graph_replans": 3,
                "max_task_replans_per_source": 0,
                "reason": "source_replan_budget_exhausted",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                NeverDispatchAdapter.instances.clear()
                graph = _graph(_task("T1"))
                planning_calls: list[dict] = []
                emitted_plans: list[dict] = []

                def blocked_plan(**kwargs) -> dict:
                    plan = {
                        "state": "blocked",
                        "task_graph_version": kwargs["task_graph_version"],
                        "graph_fingerprint": "sha256:stable-blocked-source",
                        "execution_policy": "first_unit_then_replan",
                        "units": [],
                        "dependency_edges": [],
                        "blocking": {
                            "code": "no_eligible_agent",
                            "task_ids": ["T1"],
                            "agent_inventories": {"0": [], "1": []},
                            "conflicts": ["no eligible agent"],
                            "recommended_recovery": "replan_task_graph",
                        },
                        "diagnostics": {
                            "remaining_task_ids": ["T1"],
                            "attempts": [],
                            "fallback_used": False,
                        },
                    }
                    emitted_plans.append(deepcopy(plan))
                    return plan

                def unexpected_planning(**kwargs) -> dict:
                    planning_calls.append(deepcopy(kwargs))
                    raise AssertionError("replan budget exhaustion called Planning")

                with tempfile.TemporaryDirectory() as directory:
                    args = _args(Path(directory), object())
                    args.max_task_graph_replans = case["max_task_graph_replans"]
                    args.max_task_replans_per_source = case[
                        "max_task_replans_per_source"
                    ]
                    with _patch_loop_boundaries(
                        self.module,
                        graph=graph,
                        build_execution_plan=blocked_plan,
                        adapter_type=NeverDispatchAdapter,
                        replan_from_execution_report=unexpected_planning,
                    ):
                        with self.assertRaises(
                            self.module.TaskAllocationError
                        ) as captured:
                            self.module.run_hybrid_loop(args)

                self.assertEqual(
                    captured.exception.code,
                    "allocation_replan_budget_exhausted",
                )
                self.assertEqual(planning_calls, [])
                self.assertEqual(len(emitted_plans), 1)
                diagnostics = captured.exception.diagnostics
                self.assertEqual(diagnostics["status"], "blocked")
                self.assertEqual(diagnostics["stage"], "allocation_replanning")
                self.assertEqual(
                    diagnostics["execution_plan"]["blocking"]["code"],
                    "no_eligible_agent",
                )
                replan_result = diagnostics["replan_result"]
                self.assertEqual(replan_result["status"], "skipped")
                self.assertEqual(replan_result["reason"], case["reason"])
                self.assertEqual(replan_result["trigger"], "allocation_blocked")
                self.assertEqual(
                    replan_result["blocking_budget_key"],
                    self.module.allocation_blocking_budget_key(emitted_plans[0]),
                )
                self.assertTrue(
                    replan_result["blocking_budget_key"].startswith(
                        "v1:allocation:"
                    )
                )
                self.assertEqual(len(NeverDispatchAdapter.instances), 1)
                self.assertEqual(NeverDispatchAdapter.instances[0].sent, [])
                self.assertEqual(NeverDispatchAdapter.instances[0].received, [])

    def test_each_loop_replans_and_dispatches_only_its_first_unit(self) -> None:
        graph = _graph(_task("T1"), _task("T2", ("T1",)))
        planning_chat = object()
        plan_calls: list[dict] = []

        def two_stage_plan(**kwargs) -> dict:
            plan_calls.append(deepcopy(kwargs))
            call_index = len(plan_calls)
            if call_index == 1:
                units = [
                    {
                        "time_step": 1,
                        "assignments": [{"task_id": "T1", "agent_id": "0"}],
                    },
                    {
                        "time_step": 2,
                        "assignments": [{"task_id": "T2", "agent_id": "0"}],
                    },
                ]
            elif call_index == 2:
                units = [
                    {
                        "time_step": 1,
                        "assignments": [{"task_id": "T2", "agent_id": "1"}],
                    }
                ]
            else:
                raise AssertionError("hybrid allocated more than once per execution loop")
            return {
                "state": "dispatchable",
                "task_graph_version": kwargs["task_graph_version"],
                "graph_fingerprint": f"sha256:loop-{call_index}",
                "execution_policy": "first_unit_then_replan",
                "units": units,
                "dependency_edges": [],
                "blocking": None,
                "diagnostics": {"attempts": [], "fallback_used": False},
            }

        with tempfile.TemporaryDirectory() as directory:
            args = _args(Path(directory), planning_chat)
            with _patch_loop_boundaries(
                self.module,
                graph=graph,
                build_execution_plan=two_stage_plan,
                adapter_type=RecordingAdapter,
                replan_from_execution_report=lambda **unused: {
                    "status": "failed",
                    "reason": "unexpected replan",
                },
            ):
                result = self.module.run_hybrid_loop(args)

        self.assertTrue(result["all_done"])
        self.assertEqual(result["completed_tasks"], ["T1", "T2"])
        self.assertEqual(len(plan_calls), 2)
        self.assertEqual(plan_calls[0]["progress"]["completed_task_ids"], [])
        self.assertEqual(plan_calls[1]["progress"]["completed_task_ids"], ["T1"])
        self.assertEqual(
            {task["id"] for task in plan_calls[1]["task_graph"]["flat_tasks"]},
            {"T1", "T2"},
        )
        self.assertEqual(plan_calls[1]["agent_states"][0]["observation_epoch"], 1)

        self.assertEqual(len(RecordingAdapter.instances), 1)
        sent = RecordingAdapter.instances[0].sent
        self.assertEqual(len(sent), 2)
        self.assertEqual(
            [item["subtask"]["id"] for item in sent[0]["assignments"]],
            ["T1"],
        )
        self.assertEqual(
            [item["agent_id"] for item in sent[0]["assignments"]],
            ["0"],
        )
        self.assertEqual(
            [item["subtask"]["id"] for item in sent[1]["assignments"]],
            ["T2"],
        )
        self.assertEqual(
            [item["agent_id"] for item in sent[1]["assignments"]],
            ["1"],
        )
        self.assertEqual(len(sent[0]["execution_plan"]["units"]), 2)
        self.assertEqual(len(sent[1]["execution_plan"]["units"]), 1)


if __name__ == "__main__":
    unittest.main()
