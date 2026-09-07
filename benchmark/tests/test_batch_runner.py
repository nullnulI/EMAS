from benchmark import batch_runner


def test_reset_manifest_removes_previous_batch_entries(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"episode_id": "stale"}\n', encoding="utf-8")

    batch_runner._reset_manifest(manifest)

    assert manifest.read_text(encoding="utf-8") == ""


def test_configure_runtime_environment_sets_defaults(monkeypatch, tmp_path):
    emas_root = tmp_path / "EMAS"
    conda_prefix = tmp_path / "conceptgraph"
    (conda_prefix / "lib").mkdir(parents=True)

    monkeypatch.setattr(batch_runner, "EMAS_ROOT", emas_root)
    monkeypatch.setattr(batch_runner.sys, "prefix", str(conda_prefix))
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "EMAS_PLANNING_MEMORY_CUDA_VISIBLE_DEVICES",
        "EMAS_PLANNING_DEVICE",
        "EMAS_AGENTS_CUDA_VISIBLE_DEVICES",
        "ROOT_DIR",
        "GSA_PATH",
        "MPLBACKEND",
        "HF_HUB_DISABLE_XET",
        "LD_LIBRARY_PATH",
        "VK_DRIVER_FILES",
        "VK_ICD_FILENAMES",
    ):
        monkeypatch.delenv(name, raising=False)

    batch_runner.configure_runtime_environment()

    assert batch_runner.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert batch_runner.os.environ["EMAS_PLANNING_DEVICE"] == "cuda:1"
    assert batch_runner.os.environ["EMAS_AGENTS_CUDA_VISIBLE_DEVICES"] == "1"
    assert batch_runner.os.environ["ROOT_DIR"] == str(emas_root)
    assert batch_runner.os.environ["GSA_PATH"] == str(
        emas_root / "memory" / "Grounded-Segment-Anything"
    )
    assert batch_runner.os.environ["MPLBACKEND"] == "Agg"
    assert batch_runner.os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert batch_runner.os.environ["LD_LIBRARY_PATH"] == str(conda_prefix / "lib")
    assert batch_runner.os.environ["VK_DRIVER_FILES"] == (
        "/etc/vulkan/icd.d/nvidia_icd.json"
    )
    assert batch_runner.os.environ["VK_ICD_FILENAMES"] == (
        "/etc/vulkan/icd.d/nvidia_icd.json"
    )


def test_configure_runtime_environment_replaces_generic_cuda_mask(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("GSA_PATH", "/custom/gsa")
    monkeypatch.setenv("VK_DRIVER_FILES", "/stale/mesa_icd.json")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/usr/local/cuda/compat:/stale")

    batch_runner.configure_runtime_environment()

    assert batch_runner.os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert batch_runner.os.environ["GSA_PATH"] == "/custom/gsa"
    assert batch_runner.os.environ["VK_DRIVER_FILES"] == (
        "/etc/vulkan/icd.d/nvidia_icd.json"
    )
    assert batch_runner.os.environ["LD_LIBRARY_PATH"] == (
        f"{batch_runner.sys.prefix}/lib"
    )


def test_configure_runtime_environment_allows_explicit_library_override(monkeypatch):
    monkeypatch.setenv("EMAS_LD_LIBRARY_PATH", "/custom/runtime/lib")

    batch_runner.configure_runtime_environment()

    assert batch_runner.os.environ["LD_LIBRARY_PATH"] == "/custom/runtime/lib"


def test_configure_runtime_environment_allows_separate_gpu_overrides(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("EMAS_PLANNING_MEMORY_CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("EMAS_AGENTS_CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("EMAS_PLANNING_DEVICE", "cuda:0")

    batch_runner.configure_runtime_environment()

    assert batch_runner.os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert batch_runner.os.environ["EMAS_AGENTS_CUDA_VISIBLE_DEVICES"] == "3"
    assert batch_runner.os.environ["EMAS_PLANNING_DEVICE"] == "cuda:0"


def test_kitchen_floorplan_filter_uses_official_ai2thor_range():
    assert batch_runner._is_kitchen_floorplan("FloorPlan1")
    assert batch_runner._is_kitchen_floorplan("FloorPlan30")
    assert not batch_runner._is_kitchen_floorplan("FloorPlan31")
    assert not batch_runner._is_kitchen_floorplan("FloorPlan201")
    assert not batch_runner._is_kitchen_floorplan("kitchen")


def test_batch_parser_enables_safe_model_reuse_by_default():
    args = batch_runner.build_parser().parse_args([])
    assert args.kitchen_only is False
    assert args.reuse_planning_model is True
    assert args.reuse_task_service is True
    args = batch_runner.build_parser().parse_args(["--no-reuse-planning-model", "--no-reuse-task-service"])
    assert args.reuse_planning_model is False
    assert args.reuse_task_service is False
