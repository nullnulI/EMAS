"""
Run the full ConceptGraphs pipeline from posed RGB-D data to scene graph files.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def build_gsa_variant(class_set: str, sam_variant: str, exp_suffix: str | None) -> str:
    save_name = class_set
    if sam_variant != "sam":
        save_name += f"_{sam_variant}"
    if exp_suffix:
        save_name += f"_{exp_suffix}"
    return save_name


def announce_call(name: str, *, dry_run: bool) -> bool:
    prefix = "[DRY-RUN]" if dry_run else "[CALL]"
    print(f"\n{prefix} {name}\n")
    return not dry_run


def detection_dir(scene_dir: Path, gsa_variant: str) -> Path:
    return scene_dir / f"gsa_detections_{gsa_variant}"


def infer_cachedir(scene_dir: Path, cachedir: Path | None) -> Path:
    return cachedir if cachedir is not None else scene_dir / "sg_cache"


def infer_mapfile(
    scene_dir: Path,
    gsa_variant: str,
    cfslam_save_suffix: str,
    use_post_map: bool,
    explicit_mapfile: Path | None,
) -> Path:
    if explicit_mapfile is not None:
        return explicit_mapfile

    pcd_dir = scene_dir / "pcd_saves"
    base = pcd_dir / f"full_pcd_{gsa_variant}_{cfslam_save_suffix}.pkl.gz"
    post = pcd_dir / f"full_pcd_{gsa_variant}_{cfslam_save_suffix}_post.pkl.gz"

    if use_post_map:
        if post.exists():
            return post
        if base.exists():
            return base
        return post

    if base.exists():
        return base
    if post.exists():
        return post
    return base


def file_has_payload(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def dir_has_files(path: Path, pattern: str) -> bool:
    return path.exists() and any(path.glob(pattern))


def choose_skip_bg(args: argparse.Namespace) -> bool:
    if args.skip_bg is not None:
        return args.skip_bg
    return args.class_set == "none"


def choose_class_agnostic(args: argparse.Namespace) -> bool:
    if args.class_agnostic is not None:
        return args.class_agnostic
    return args.class_set == "none"


def choose_mask_conf_threshold(args: argparse.Namespace) -> float:
    if args.mask_conf_threshold is not None:
        return args.mask_conf_threshold
    return 0.95 if args.class_set == "none" else 0.25


def ensure_required_inputs(args: argparse.Namespace, gsa_output_dir: Path) -> None:
    if not args.skip_gsa and "GSA_PATH" not in os.environ:
        raise RuntimeError("GSA_PATH is not set. generate_gsa_results.py requires it.")

    if args.skip_gsa and not args.skip_cfslam and not dir_has_files(gsa_output_dir, "*.pkl.gz"):
        raise FileNotFoundError(
            "GSA stage is skipped, but no existing detection cache was found at "
            f"{gsa_output_dir}"
        )

    if args.skip_cfslam and not (args.skip_extract and args.skip_refine and args.skip_build):
        if not file_has_payload(args.mapfile):
            raise FileNotFoundError(f"Map file not found: {args.mapfile}")

    if (not args.skip_refine or not args.skip_build) and args.vlm_backend == "qwen":
        if args.vlm_model_path is None and os.getenv("QWEN2_5_VL_MODEL_PATH") is None:
            raise RuntimeError(
                "Qwen backend selected, but no local model path was provided. "
                "Pass --vlm_model_path or set QWEN2_5_VL_MODEL_PATH."
            )


def run_gsa_stage(args: argparse.Namespace) -> None:
    from conceptgraph.scripts import generate_gsa_results

    # Call the Python entrypoint directly in-process. This intentionally avoids
    # spawning `python generate_gsa_results.py ...`, while keeping its cleanup.
    gsa_args = argparse.Namespace(
        dataset_root=args.dataset_root,
        dataset_config=str(args.dataset_config),
        scene_id=args.scene_id,
        start=args.start,
        end=args.end,
        stride=args.stride,
        desired_height=args.desired_height,
        desired_width=args.desired_width,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        nms_threshold=args.nms_threshold,
        class_set=args.class_set,
        detector=args.detector,
        add_bg_classes=args.add_bg_classes,
        accumu_classes=args.accumu_classes,
        sam_variant=args.sam_variant,
        save_video=False,
        device=args.device,
        use_slow_vis=False,
        exp_suffix=args.gsa_exp_suffix,
    )
    generate_gsa_results.main(gsa_args)


def build_cfslam_cfg(
    args: argparse.Namespace,
    *,
    gsa_variant: str,
    mask_conf_threshold: float,
    skip_bg: bool,
    class_agnostic: bool,
):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(REPO_ROOT / "conceptgraph" / "configs" / "slam_pipeline" / "base.yaml")
    cfg.dataset_root = str(args.dataset_root)
    cfg.dataset_config = str(args.dataset_config)
    cfg.scene_id = args.scene_id
    cfg.start = args.start
    cfg.end = args.end
    cfg.stride = args.stride
    cfg.image_height = args.desired_height
    cfg.image_width = args.desired_width
    cfg.device = args.device
    cfg.gsa_variant = gsa_variant
    cfg.detection_folder_name = f"gsa_detections_{gsa_variant}"
    cfg.det_vis_folder_name = f"gsa_vis_{gsa_variant}"
    cfg.color_file_name = f"gsa_classes_{gsa_variant}"
    cfg.spatial_sim_type = args.spatial_sim_type
    cfg.match_method = args.match_method
    cfg.sim_threshold = args.sim_threshold
    cfg.mask_conf_threshold = mask_conf_threshold
    cfg.obj_min_detections = args.obj_min_detections
    cfg.dbscan_eps = args.dbscan_eps
    cfg.skip_bg = skip_bg
    cfg.class_agnostic = class_agnostic
    cfg.max_bbox_area_ratio = args.max_bbox_area_ratio
    cfg.save_suffix = args.cfslam_save_suffix
    cfg.merge_interval = args.merge_interval
    cfg.merge_visual_sim_thresh = args.merge_visual_sim_thresh
    cfg.merge_text_sim_thresh = args.merge_text_sim_thresh
    cfg.denoise_interval = args.denoise_interval
    cfg.filter_interval = args.filter_interval
    cfg.return_in_memory_map = True
    return cfg


def run_cfslam_stage(
    args: argparse.Namespace,
    *,
    gsa_variant: str,
    mask_conf_threshold: float,
    skip_bg: bool,
    class_agnostic: bool,
) -> dict[str, Any] | None:
    from conceptgraph.slam import cfslam_pipeline_batch

    cfg = build_cfslam_cfg(
        args,
        gsa_variant=gsa_variant,
        mask_conf_threshold=mask_conf_threshold,
        skip_bg=skip_bg,
        class_agnostic=class_agnostic,
    )
    return cfslam_pipeline_batch.main(cfg)


def make_scenegraph_args(args: argparse.Namespace, *, cachedir: Path, mapfile: Path, mode: str):
    from conceptgraph.scenegraph.build_scenegraph_cfslam import ProgramArgs

    return ProgramArgs(
        mode=mode,
        cachedir=str(cachedir),
        mapfile=str(mapfile),
        device=args.scenegraph_device,
        masking_option=args.masking_option,
        max_detections_per_object=args.max_detections_per_object,
        min_views_per_object=args.min_views_per_object,
        vlm_backend=args.vlm_backend,
        vlm_model_path=str(args.vlm_model_path) if args.vlm_model_path is not None else None,
        vlm_conv_mode=args.vlm_conv_mode,
        vlm_num_gpus=args.vlm_num_gpus,
    )


def build_shared_vlm_chat(args: argparse.Namespace, *, need_image_captioning: bool, need_local_text: bool):
    if not need_image_captioning and not need_local_text:
        return None

    from conceptgraph.vlm import build_vlm_chat

    chat = build_vlm_chat(
        backend=args.vlm_backend,
        model_path=str(args.vlm_model_path) if args.vlm_model_path is not None else None,
        conv_mode=args.vlm_conv_mode,
        num_gpus=args.vlm_num_gpus,
    )
    print(f"{args.vlm_backend} chat initialized once for scenegraph stages.")
    return chat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run generate_gsa_results, cfslam_pipeline_batch, and build_scenegraph_cfslam end-to-end."
    )

    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--dataset_config", type=Path, required=True)
    parser.add_argument("--scene_id", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--stride", type=int, default=5)

    parser.add_argument("--desired_height", type=int, default=480)
    parser.add_argument("--desired_width", type=int, default=640)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--scenegraph_device", type=str, default="cuda:0")

    parser.add_argument(
        "--class_set",
        type=str,
        default="ram",
        choices=["scene", "generic", "minimal", "tag2text", "ram", "none"],
    )
    parser.add_argument("--detector", type=str, default="dino", choices=["yolo", "dino"])
    parser.add_argument(
        "--sam_variant",
        type=str,
        default="sam",
        choices=["sam", "fastsam", "mobilesam", "lighthqsam"],
    )
    parser.add_argument("--box_threshold", type=float, default=0.2)
    parser.add_argument("--text_threshold", type=float, default=0.2)
    parser.add_argument("--nms_threshold", type=float, default=0.5)
    parser.add_argument("--add_bg_classes", action="store_true")
    parser.add_argument("--accumu_classes", action="store_true")
    parser.add_argument("--gsa_exp_suffix", type=str, default=None)
    parser.add_argument(
        "--gsa_variant",
        type=str,
        default=None,
        help="Optional manual override for the generated GSA result folder suffix.",
    )

    parser.add_argument("--spatial_sim_type", type=str, default="overlap")
    parser.add_argument("--match_method", type=str, default="sim_sum")
    parser.add_argument("--sim_threshold", type=float, default=1.2)
    parser.add_argument("--mask_conf_threshold", type=float, default=None)
    parser.add_argument("--obj_min_detections", type=int, default=1)
    parser.add_argument("--dbscan_eps", type=float, default=0.1)
    parser.add_argument("--max_bbox_area_ratio", type=float, default=0.5)
    parser.add_argument("--merge_interval", type=int, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=0.8)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=0.8)
    parser.add_argument("--denoise_interval", type=int, default=20)
    parser.add_argument("--filter_interval", type=int, default=-1)
    parser.add_argument("--cfslam_save_suffix", type=str, default="overlap_maskconf0.25_simsum1.2_dbscan.1")
    parser.add_argument("--skip_bg", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--class_agnostic", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--cachedir", type=Path, default=None)
    parser.add_argument("--mapfile", type=Path, default=None)
    parser.add_argument("--use_post_map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--masking_option", type=str, default="none", choices=["blackout", "red_outline", "none"])
    parser.add_argument("--max_detections_per_object", type=int, default=10)
    parser.add_argument("--min_views_per_object", type=int, default=1)
    parser.add_argument("--vlm_backend", type=str, default="qwen", choices=["llava", "qwen"])
    parser.add_argument("--vlm_model_path", type=Path, default=None)
    parser.add_argument("--vlm_conv_mode", type=str, default="v0_mmtag")
    parser.add_argument("--vlm_num_gpus", type=int, default=1)

    parser.add_argument("--skip_gsa", action="store_true")
    parser.add_argument("--skip_cfslam", action="store_true")
    parser.add_argument("--skip_extract", action="store_true")
    parser.add_argument("--skip_refine", action="store_true")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--generate_scenegraph_json", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")

    return parser


def build_pipeline_args(
    *,
    dataset_root: Path,
    dataset_config: Path,
    scene_id: str,
    **overrides: Any,
) -> argparse.Namespace:
    """Build an in-process pipeline configuration with CLI-compatible defaults."""
    args = build_parser().parse_args(
        [
            "--dataset_root",
            str(dataset_root),
            "--dataset_config",
            str(dataset_config),
            "--scene_id",
            scene_id,
        ]
    )
    unknown = sorted(set(overrides) - vars(args).keys())
    if unknown:
        raise TypeError(f"Unknown pipeline argument(s): {', '.join(unknown)}")
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def run_pipeline(args: argparse.Namespace) -> dict[str, Path | None]:
    """Run the full pipeline in-process and return its expected output paths."""
    scene_dir = args.dataset_root / args.scene_id
    gsa_variant = args.gsa_variant or build_gsa_variant(args.class_set, args.sam_variant, args.gsa_exp_suffix)
    mask_conf_threshold = choose_mask_conf_threshold(args)
    skip_bg = choose_skip_bg(args)
    class_agnostic = choose_class_agnostic(args)
    cachedir = infer_cachedir(scene_dir, args.cachedir)
    mapfile = infer_mapfile(scene_dir, gsa_variant, args.cfslam_save_suffix, args.use_post_map, args.mapfile)

    args.mapfile = mapfile
    cachedir.mkdir(parents=True, exist_ok=True)

    print("Pipeline configuration:")
    print(f"  repo_root: {REPO_ROOT}")
    print(f"  scene_dir: {scene_dir}")
    print(f"  gsa_variant: {gsa_variant}")
    print(f"  cachedir: {cachedir}")
    print(f"  mapfile: {mapfile}")

    gsa_output_dir = detection_dir(scene_dir, gsa_variant)
    ensure_required_inputs(args, gsa_output_dir)
    captions_file = cachedir / "cfslam_node_captions.json"
    refine_dir = cachedir / "cfslam_gpt-4_responses"
    relations_file = cachedir / "cfslam_object_relations.json"
    edges_file = cachedir / "cfslam_scenegraph_edges.pkl"
    pruned_mapfile = cachedir / "map" / "scene_map_cfslam_pruned.pkl.gz"
    scenegraph_json = cachedir / "scene_graph.json"

    run_extract = not args.skip_extract and not (args.skip_existing and file_has_payload(captions_file))
    run_refine = not args.skip_refine and not (args.skip_existing and dir_has_files(refine_dir, "*.json"))
    run_build = not args.skip_build and not (
        args.skip_existing and file_has_payload(edges_file) and file_has_payload(relations_file)
    )
    run_generate_json = args.generate_scenegraph_json and not (
        args.skip_existing and file_has_payload(scenegraph_json)
    )

    if not args.skip_gsa:
        if args.skip_existing and dir_has_files(gsa_output_dir, "*.pkl.gz"):
            print(f"[SKIP] GSA detections already exist: {gsa_output_dir}")
        else:
            if announce_call("direct Python call: generate_gsa_results.main(args)", dry_run=args.dry_run):
                run_gsa_stage(args)

    cfslam_result = None
    scene_map_in_memory = None
    scene_map_data = None
    if not args.skip_cfslam:
        if args.skip_existing and file_has_payload(mapfile):
            print(f"[SKIP] CFSLAM map already exists: {mapfile}")
        else:
            if announce_call("cfslam_pipeline_batch.main(cfg)", dry_run=args.dry_run):
                cfslam_result = run_cfslam_stage(
                    args,
                    gsa_variant=gsa_variant,
                    mask_conf_threshold=mask_conf_threshold,
                    skip_bg=skip_bg,
                    class_agnostic=class_agnostic,
                )
                if isinstance(cfslam_result, dict):
                    scene_map_in_memory = cfslam_result.pop("scene_map", None)
                    scene_map_data = cfslam_result.get("post") or cfslam_result.get("pre")
                    cfslam_result = None

    if args.dry_run:
        print("\n[DRY-RUN] Scenegraph stages would be called directly with shared in-process state.")

    scenegraph_args = None
    scene_map = scene_map_in_memory
    captions = None
    refined_responses = None
    build_result = None
    shared_chat = None

    if not args.dry_run:
        from conceptgraph.scenegraph import build_scenegraph_cfslam as scenegraph

        scenegraph_args = make_scenegraph_args(
            args,
            cachedir=cachedir,
            mapfile=mapfile,
            mode="extract-node-captions",
        )
        need_scene_map = run_extract or run_refine or run_build
        if need_scene_map and scene_map is None:
            scene_map = scenegraph.load_scene_map_from_data(scenegraph_args, scene_map_data)

        need_local_text = (run_refine or run_build) and scenegraph.should_use_local_refinement(scenegraph_args)
        shared_chat = build_shared_vlm_chat(
            scenegraph_args,
            need_image_captioning=run_extract,
            need_local_text=need_local_text,
        )

    try:
        if not args.skip_extract:
            if not run_extract:
                print(f"[SKIP] Caption cache already exists: {captions_file}")
            elif announce_call("build_scenegraph_cfslam.extract_node_captions(...)", dry_run=args.dry_run):
                assert scenegraph_args is not None
                scenegraph_args.mode = "extract-node-captions"
                captions = scenegraph.extract_node_captions(scenegraph_args, scene_map=scene_map, chat=shared_chat)

        if not args.skip_refine:
            if not run_refine:
                print(f"[SKIP] Refined captions already exist: {refine_dir}")
            elif announce_call("build_scenegraph_cfslam.refine_node_captions(...)", dry_run=args.dry_run):
                assert scenegraph_args is not None
                scenegraph_args.mode = "refine-node-captions"
                refined_responses = scenegraph.refine_node_captions(
                    scenegraph_args, scene_map=scene_map, captions=captions, chat=shared_chat
                )

        if not args.skip_build:
            if not run_build:
                print(f"[SKIP] Scenegraph files already exist in: {cachedir}")
            elif announce_call("build_scenegraph_cfslam.build_scenegraph(...)", dry_run=args.dry_run):
                assert scenegraph_args is not None
                scenegraph_args.mode = "build-scenegraph"
                build_result = scenegraph.build_scenegraph(
                    scenegraph_args, scene_map=scene_map, refined_responses=refined_responses, chat=shared_chat
                )

        if args.generate_scenegraph_json:
            if not run_generate_json:
                print(f"[SKIP] scene_graph.json already exists: {scenegraph_json}")
            elif announce_call("build_scenegraph_cfslam.generate_scenegraph_json(...)", dry_run=args.dry_run):
                assert scenegraph_args is not None
                scenegraph_args.mode = "generate-scenegraph-json"
                pruned_scene_map = build_result["scene_map"] if build_result is not None else None
                scenegraph.generate_scenegraph_json(scenegraph_args, scene_map=pruned_scene_map)
    finally:
        if shared_chat is not None:
            from conceptgraph.vlm import close_vlm_chat

            close_vlm_chat(shared_chat)
            shared_chat = None

    print("\nExpected outputs:")
    for path in [
        gsa_output_dir,
        mapfile,
        captions_file,
        refine_dir,
        cachedir / "cfslam_object_relation_queries.json",
        relations_file,
        edges_file,
        pruned_mapfile,
        scenegraph_json if args.generate_scenegraph_json else None,
    ]:
        if path is not None:
            print(f"  - {path}")

    return {
        "cachedir": cachedir,
        "mapfile": mapfile,
        "captions": captions_file,
        "relations": edges_file,
        "object_relations": relations_file,
        "pruned_mapfile": pruned_mapfile,
        "scene_graph": scenegraph_json if args.generate_scenegraph_json else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
