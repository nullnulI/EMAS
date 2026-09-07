from benchmark.config_loader import load_episodes
from benchmark.official import resolve_mapthor_root


def test_vendored_assets_are_discoverable():
    root = resolve_mapthor_root()
    assert root.name == "mapthor_assets"
    assert (root / "AI2Thor" / "Tasks").is_dir()


def test_load_known_type1_episode():
    root = resolve_mapthor_root()
    episodes = load_episodes(
        root / "configs",
        task_ids=[1],
        floorplan_indices=[0],
        seeds=[0],
        agent_count=2,
        tasks_root=root / "AI2Thor" / "Tasks",
    )
    assert len(episodes) == 1
    assert episodes[0].task_name == "1_put_bread_lettuce_tomato_fridge"
    assert episodes[0].floorplan == "FloorPlan1"
    assert episodes[0].timeout == 30
