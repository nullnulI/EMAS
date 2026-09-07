from __future__ import annotations

import atexit
import json
import gc
import os
import re
import sys
from copy import deepcopy
from pathlib import Path


EMAS_ROOT = Path(__file__).resolve().parents[1]


def release_episode_memory() -> None:
    """Release unreachable episode state and unused CUDA allocator blocks."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is not None:
            malloc_trim(0)
    except (OSError, TypeError):
        pass


def configure_runtime_environment() -> None:
    # Memory uses cuda:0 while planning is pinned to cuda:1. Agent models run in
    # a managed subprocess whose visibility mask exposes physical GPU 1 only.
    planning_memory_devices = os.environ.get(
        "EMAS_PLANNING_MEMORY_CUDA_VISIBLE_DEVICES"
    )
    if planning_memory_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = planning_memory_devices
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ.setdefault("EMAS_PLANNING_DEVICE", "cuda:1")
    os.environ.setdefault("EMAS_AGENTS_CUDA_VISIBLE_DEVICES", "1")
    os.environ.setdefault("ROOT_DIR", str(EMAS_ROOT))
    os.environ.setdefault(
        "GSA_PATH",
        str(EMAS_ROOT / "memory" / "Grounded-Segment-Anything"),
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    nvidia_icd = Path(
        os.environ.get("EMAS_VULKAN_ICD", "/etc/vulkan/icd.d/nvidia_icd.json")
    )
    if nvidia_icd.exists():
        # VK_DRIVER_FILES takes precedence in newer Vulkan loaders, so keep both
        # variables aligned even when the parent shell contains stale settings.
        os.environ["VK_DRIVER_FILES"] = str(nvidia_icd)
        os.environ["VK_ICD_FILENAMES"] = str(nvidia_icd)

    conda_lib = Path(sys.prefix) / "lib"
    if conda_lib.is_dir():
        # Do not inherit CUDA compat/stub paths from an interactive shell. They
        # can make NVIDIA's Vulkan ICD load a libcuda version from another driver.
        os.environ["LD_LIBRARY_PATH"] = os.environ.get(
            "EMAS_LD_LIBRARY_PATH", str(conda_lib)
        )


configure_runtime_environment()

from benchmark.config_loader import load_episodes
from benchmark.metrics import aggregate_results
from benchmark.official import resolve_mapthor_root
from scripts import benchmark_hybrid_decision_loop as benchmark_hybrid


def _csv_ints(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _is_kitchen_floorplan(floorplan: str) -> bool:
    """AI2-THOR kitchens use the canonical FloorPlan1 through FloorPlan30 range."""
    match = re.fullmatch(r"FloorPlan(\d+)", str(floorplan))
    return bool(match and 1 <= int(match.group(1)) <= 30)


def _reset_manifest(path: Path) -> None:
    """Start a manifest for the current batch invocation."""
    path.write_text("", encoding="utf-8")


def build_parser():
    parser = benchmark_hybrid.build_parser()
    parser.description = "Batch-run the benchmark-only EMAS hybrid copy on MAP-THOR."
    for group in parser._mutually_exclusive_groups:
        destinations = {action.dest for action in group._group_actions}
        if {"mapthor_task_id", "mapthor_task_name"} & destinations:
            group.required = False
    parser.add_argument("--mapthor-task-categories", help="Comma-separated categories, e.g. 1,2")
    parser.add_argument("--mapthor-task-ids", help="Comma-separated official task IDs")
    parser.add_argument("--mapthor-floorplan-indices", default="0")
    parser.add_argument("--benchmark-seeds", default="0")
    parser.add_argument("--kitchen-only", action="store_true", help="Run only canonical AI2-THOR kitchen floorplans (FloorPlan1-30).")
    parser.add_argument("--reuse-planning-model", dest="reuse_planning_model", action="store_true", default=True)
    parser.add_argument("--no-reuse-planning-model", dest="reuse_planning_model", action="store_false")
    parser.add_argument("--reuse-task-service", dest="reuse_task_service", action="store_true", default=True)
    parser.add_argument("--no-reuse-task-service", dest="reuse_task_service", action="store_false")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # QwenChatAdapter uses this path to distinguish the planning LLM from the
    # scene-graph VLM, which remains on memory's default cuda:0 device.
    os.environ["PLANNING_MODEL_PATH"] = str(args.planning_model_path)
    mapthor_root = resolve_mapthor_root(args.mapthor_root)
    config_dir = args.mapthor_config_dir or mapthor_root / "configs"
    task_ids = _csv_ints(args.mapthor_task_ids)
    if task_ids is None and args.mapthor_task_id is not None:
        task_ids = [args.mapthor_task_id]
    episodes = load_episodes(
        config_dir,
        task_categories=_csv_ints(args.mapthor_task_categories),
        task_ids=task_ids,
        task_names=[args.mapthor_task_name] if args.mapthor_task_name else None,
        floorplan_indices=_csv_ints(args.mapthor_floorplan_indices),
        seeds=_csv_ints(args.benchmark_seeds) or [0],
        agent_count=args.agentnum,
        tasks_root=mapthor_root / "AI2Thor" / "Tasks",
        require_assets=True,
    )
    if args.kitchen_only:
        episodes = [episode for episode in episodes if _is_kitchen_floorplan(episode.floorplan)]
    if args.limit is not None:
        episodes = episodes[: max(0, args.limit)]
    if not episodes:
        raise SystemExit("No runnable episodes matched the requested filters.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    _reset_manifest(manifest_path)
    results = []
    shared_task_service_manager = None
    shared_task_service_info = None
    shared_planning_chat = None
    shared_resources_closed = False

    def close_shared_resources() -> None:
        nonlocal shared_resources_closed
        if shared_resources_closed:
            return
        shared_resources_closed = True
        if shared_planning_chat is not None:
            try:
                from conceptgraph.vlm import close_vlm_chat
                close_vlm_chat(shared_planning_chat)
            except Exception as exc:
                print(f"warning: shared planning model cleanup failed: {exc!r}", flush=True)
        if shared_task_service_manager is not None:
            try:
                shared_task_service_manager.__exit__(None, None, None)
            except Exception as exc:
                print(f"warning: shared task service cleanup failed: {exc!r}", flush=True)

    atexit.register(close_shared_resources)

    if args.reuse_task_service and args.execution_mode == "task_service" and args.task_service_autostart:
        try:
            receiver_port = benchmark_hybrid._find_available_port(int(getattr(args, "receiver_port_base", 19010)))
            args.receiver_port_base = receiver_port
            receiver_url = f"http://127.0.0.1:{receiver_port}"
            shared_root = output_dir / "_shared_services"
            shared_task_service_manager = benchmark_hybrid._managed_task_execution_server(
                args,
                shared_root,
                receiver_url=receiver_url,
                wait_for_receiver=False,
            )
            shared_task_service_info = dict(shared_task_service_manager.__enter__())
            shared_task_service_info["receiver_url"] = receiver_url
            print(f"[batch] task service model loaded once at {shared_task_service_info['task_service_url']}", flush=True)
        except Exception as exc:
            print(f"warning: shared task service startup failed; falling back to per-episode service: {exc!r}", flush=True)
            shared_task_service_manager = None
            shared_task_service_info = None

    if args.reuse_planning_model and not args.disable_qwen:
        try:
            from conceptgraph.vlm import build_vlm_chat
            shared_planning_chat = build_vlm_chat(
                backend="qwen",
                model_path=args.planning_model_path,
                conv_mode=args.qwen_conv_mode,
                num_gpus=args.qwen_num_gpus,
            )
            print(f"[batch] planning model loaded once from {args.planning_model_path}", flush=True)
        except Exception as exc:
            print(f"warning: shared planning model startup failed; falling back to per-episode loading: {exc!r}", flush=True)

    for index, episode in enumerate(episodes, start=1):
        result_path = episode.result_dir(output_dir) / "benchmark_result.json"
        if result_path.exists() and not args.rerun:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            print(f"[{index}/{len(episodes)}] reuse {episode.episode_id}", flush=True)
        else:
            print(f"[{index}/{len(episodes)}] run {episode.episode_id}", flush=True)
            episode_args = deepcopy(args)
            episode_args.seed = episode.seed
            episode_args.agentnum = episode.agent_count
            episode_args.run_name = episode.episode_id
            if shared_planning_chat is not None:
                episode_args._shared_planning_chat = shared_planning_chat
            if shared_task_service_info is not None:
                episode_args._shared_task_service_info = shared_task_service_info
                episode_args._task_service_namespace = episode.episode_id
            result = benchmark_hybrid.run_benchmark_episode(
                episode_args, episode, mapthor_root=mapthor_root
            )
        # The full hybrid record is already persisted in benchmark_result.json.
        # Keeping it for every episode makes batch-process RAM grow linearly.
        results.append({"evaluation": result.get("evaluation")})
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "episode_id": episode.episode_id,
                        "result": str(result_path),
                        "evaluation": result.get("evaluation"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        del result
        release_episode_memory()

    close_shared_resources()
    atexit.unregister(close_shared_resources)

    summary = aggregate_results(results)
    summary["mapthor_root"] = str(mapthor_root)
    summary["episode_ids"] = [episode.episode_id for episode in episodes]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
