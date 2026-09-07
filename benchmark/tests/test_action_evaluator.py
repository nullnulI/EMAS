from benchmark.action_log import BenchmarkActionLogger
from benchmark.evaluator import evaluate_official_checker


class FakeChecker:
    def __init__(self):
        self.subtasks = ["PickupObject(Bread)"]
        self.subtasks_completed_numerated = []
        self.coverage_completed = []

    def perform_metric_check(self, action, success, inventory):
        if action == "PickupObject(Bread_1)" and success:
            self.subtasks_completed_numerated.append(action)
            self.coverage_completed.append("Bread_1")

    def get_transport_rate(self):
        return len(self.subtasks_completed_numerated) / len(self.subtasks)

    def get_coverage(self):
        return float(len(self.coverage_completed))

    def check_success(self):
        return len(self.subtasks_completed_numerated) == len(self.subtasks)


class FakeEvent:
    metadata = {"inventoryObjects": [{"objectId": "Bread|0|0|0"}]}


def test_new_logger_clears_previous_action_log(tmp_path):
    path = tmp_path / "actions.jsonl"
    path.write_text('{"stale": true}\n', encoding="utf-8")

    BenchmarkActionLogger(path, FakeChecker(), [])

    assert path.read_text(encoding="utf-8") == ""


def test_successful_real_action_drives_checker(tmp_path):
    checker = FakeChecker()
    logger = BenchmarkActionLogger(
        tmp_path / "actions.jsonl",
        checker,
        [{"objectId": "Bread|0|0|0", "objectType": "Bread"}],
    )
    logger.bind(macro_step=1, subtask_id="T1", agent_id=0)
    logger(
        FakeEvent(),
        {
            "action": "PickupObject",
            "params": {"objectId": "Bread|0|0|0"},
            "lastActionSuccess": True,
        },
    )
    result = evaluate_official_checker(
        checker, logger, internal_all_done=True, macro_steps=1, timeout=30
    )
    assert result["success"] == 1
    assert result["transport_rate"] == 1.0
    assert result["progress_actions_by_agent"] == {"0": 1}


def test_logger_prefers_actual_robot_id_from_relay_callback(tmp_path):
    checker = FakeChecker()
    logger = BenchmarkActionLogger(tmp_path / "actions.jsonl", checker, [])
    logger.bind(macro_step=1, subtask_id="T1", agent_id=0)

    logger(
        FakeEvent(),
        {
            "action": "RotateRight",
            "robot_id": 1,
            "params": {},
            "lastActionSuccess": True,
        },
    )

    assert logger.records[0]["agent_id"] == 1


def test_pickup_falls_back_to_official_pick_name_when_checker_has_no_alias(tmp_path):
    class Checker:
        subtasks = []
        subtasks_completed = []

        def __init__(self):
            self.actions = []

        def perform_metric_check(self, action, success, inventory):
            self.actions.append((action, success, inventory))

    checker = Checker()
    logger = BenchmarkActionLogger(
        tmp_path / "actions.jsonl",
        checker,
        [{"objectId": "Bread|0|0|0", "objectType": "Bread"}],
    )
    logger(
        FakeEvent(),
        {
            "action": "PickupObject",
            "params": {"objectId": "Bread|0|0|0"},
            "lastActionSuccess": True,
        },
    )

    assert checker.actions == [("PickObject(Bread_1)", True, "Bread_1")]


def test_pickup_prefers_pickup_camelcase_exact_checker_alias(tmp_path):
    class Checker:
        subtasks = ["PickUpObject(Bread_1)"]
        subtasks_completed = []

        def __init__(self):
            self.actions = []

        def perform_metric_check(self, action, success, inventory):
            self.actions.append((action, success, inventory))

    checker = Checker()
    logger = BenchmarkActionLogger(
        tmp_path / "actions.jsonl",
        checker,
        [{"objectId": "Bread|0|0|0", "objectType": "Bread"}],
    )
    logger(
        FakeEvent(),
        {
            "action": "PickupObject",
            "params": {"objectId": "Bread|0|0|0"},
            "lastActionSuccess": True,
        },
    )

    assert checker.actions == [("PickUpObject(Bread_1)", True, "Bread_1")]


def test_pickup_uses_pickup_camelcase_family_prefix_for_numbered_objects(tmp_path):
    class Checker:
        subtasks = ["PickUpObject(Bread)"]
        subtasks_completed = []

        def __init__(self):
            self.actions = []

        def perform_metric_check(self, action, success, inventory):
            self.actions.append((action, success, inventory))

    checker = Checker()
    logger = BenchmarkActionLogger(
        tmp_path / "actions.jsonl",
        checker,
        [{"objectId": "Bread|0|0|0", "objectType": "Bread"}],
    )
    logger(
        FakeEvent(),
        {
            "action": "PickupObject",
            "params": {"objectId": "Bread|0|0|0"},
            "lastActionSuccess": True,
        },
    )

    assert checker.actions == [("PickUpObject(Bread_1)", True, "Bread_1")]


def test_pickup_exact_checker_alias_wins_over_family_prefix(tmp_path):
    class Checker:
        subtasks = ["PickupObject(Bread)", "PickUpObject(Bread_1)"]
        subtasks_completed = []

        def __init__(self):
            self.actions = []

        def perform_metric_check(self, action, success, inventory):
            self.actions.append((action, success, inventory))

    checker = Checker()
    logger = BenchmarkActionLogger(
        tmp_path / "actions.jsonl",
        checker,
        [{"objectId": "Bread|0|0|0", "objectType": "Bread"}],
    )
    logger(
        FakeEvent(),
        {
            "action": "PickupObject",
            "params": {"objectId": "Bread|0|0|0"},
            "lastActionSuccess": True,
        },
    )

    assert checker.actions == [("PickUpObject(Bread_1)", True, "Bread_1")]

