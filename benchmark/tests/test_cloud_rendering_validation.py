from argparse import Namespace
from subprocess import CompletedProcess

from planning.datagen import generation


def test_cloud_rendering_validation_retries_transient_failure(monkeypatch):
    results = iter(
        [
            CompletedProcess([], 1, "ERROR_INCOMPATIBLE_DRIVER"),
            CompletedProcess([], 1, "ERROR_INCOMPATIBLE_DRIVER"),
            CompletedProcess([], 0, "deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU"),
        ]
    )
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return next(results)

    monkeypatch.setattr(generation.subprocess, "run", fake_run)
    monkeypatch.setattr(generation.time, "sleep", lambda _: None)

    generation.validate_cloud_rendering_environment(Namespace(platform="cloud"))

    assert len(calls) == 3
